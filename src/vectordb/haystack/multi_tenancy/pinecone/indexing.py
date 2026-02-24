"""Pinecone multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from haystack import Document

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.multi_tenancy.common.config import load_config
from vectordb.haystack.multi_tenancy.common.embeddings import (
    create_document_embedder,
    truncate_embeddings,
)
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.timing import Timer
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


logger = logging.getLogger(__name__)


class PineconeMultitenancyIndexingPipeline:
    """Pinecone indexing pipeline with namespace tenant isolation.

    Pinecone provides multi-tenancy through:
    - One namespace per tenant
    - Namespace-based isolation
    - Automatic tenant separation
    """

    def __init__(
        self,
        config_or_path: dict[str, Any] | str | Path,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize Pinecone indexing pipeline.

        Args:
            config_or_path: Config dict or path to YAML file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self.config = load_config(config_or_path)
        self.tenant_context = TenantContext.resolve(tenant_context, self.config)
        self._index = None
        self._embedder = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Pinecone and initialize embedder."""
        from pinecone import Pinecone, ServerlessSpec

        pinecone_config = self.config.get("pinecone", self.config.get("database", {}))

        api_key = pinecone_config.get("api_key") or os.environ.get("PINECONE_API_KEY")

        self._pc = Pinecone(api_key=api_key)

        index_name = pinecone_config.get(
            "index", os.environ.get("PINECONE_INDEX", "multitenancy")
        )

        existing_indexes = [idx.name for idx in self._pc.list_indexes()]
        if index_name not in existing_indexes:
            dimension = self.config.get("embedding", {}).get("dimension", 1024)
            metric = pinecone_config.get("metric", "cosine")
            cloud = pinecone_config.get("cloud", "aws")
            region = pinecone_config.get("region", "us-east-1")

            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self._index = self._pc.Index(index_name)

        self._embedder = create_document_embedder(self.config)
        self._embedder.warm_up()

    def _get_index_name(self) -> str:
        """Get index name from config."""
        pinecone_config = self.config.get("pinecone", self.config.get("database", {}))
        return pinecone_config.get(
            "index", os.environ.get("PINECONE_INDEX", "multitenancy")
        )

    def close(self) -> None:
        """Close Pinecone connection."""
        # Pinecone doesn't have a specific close method, but we can reset the connection
        pass

    def _load_documents_from_dataloader(self) -> list[Document]:
        """Load documents from configured dataloader.

        Returns:
            List of Haystack Documents loaded from dataloader.
        """
        dataloader_config = self.config.get("dataloader", {})
        dataset_type = dataloader_config.get("dataset", "triviaqa")
        params = dataloader_config.get("params", {})

        dataset_id = params.get("dataset_name")
        split = params.get("split", "test")
        limit = params.get("limit")

        loader = DataloaderCatalog.create(
            dataset_type,
            split=split,
            limit=limit,
            dataset_id=dataset_id,
        )

        return loader.load().to_haystack()

    def run(
        self,
        documents: list[Document] | None = None,
        tenant_id: str | None = None,
    ) -> TenantIndexResult:
        """Index documents for a tenant.

        Args:
            documents: Documents to index. If None, loads from dataloader.
            tenant_id: Target tenant. Uses context if None.

        Returns:
            TenantIndexResult with indexing metrics.
        """
        target_tenant = tenant_id or self.tenant_context.tenant_id

        with Timer() as timer:
            if documents is None:
                documents = self._load_documents_from_dataloader()

            if not documents:
                logger.warning("No documents to index")
                return TenantIndexResult(
                    tenant_id=target_tenant,
                    documents_indexed=0,
                    collection_name=self._get_index_name(),
                    timing=self._create_timing_metrics(0, timer.elapsed_ms),
                )

            # Embed documents
            result = self._embedder.run(documents=documents)
            embedded_docs = result["documents"]

            # Truncate if output_dimension specified
            output_dim = self.config.get("embedding", {}).get("output_dimension")
            embedded_docs = truncate_embeddings(embedded_docs, output_dim)

            # Prepare vectors for Pinecone with tenant namespace
            vectors = []
            for i, doc in enumerate(embedded_docs):
                if doc.embedding is not None:
                    doc_id = f"{target_tenant}_{i}"

                    # Prepare metadata
                    metadata = doc.meta.copy()
                    metadata["tenant_id"] = target_tenant
                    metadata["content"] = doc.content or ""

                    vectors.append((doc_id, doc.embedding, metadata))

            # Upsert vectors to Pinecone with tenant namespace
            if vectors:
                # Split into batches of 100 (Pinecone recommended batch size)
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i : i + batch_size]
                    self._index.upsert(
                        vectors=batch,
                        namespace=target_tenant,  # This is the key for Pinecone multi-tenancy
                    )

        metrics = self._create_timing_metrics(len(embedded_docs), timer.elapsed_ms)

        result = TenantIndexResult(
            tenant_id=target_tenant,
            documents_indexed=len(embedded_docs),
            collection_name=self._get_index_name(),
            timing=metrics,
        )

        logger.info(
            "Indexed %d documents for tenant %s in %.1fms",
            len(embedded_docs),
            target_tenant,
            timer.elapsed_ms,
        )

        return result

    def _create_timing_metrics(
        self,
        num_documents: int,
        total_ms: float,
    ) -> Any:
        """Create timing metrics for indexing operation.

        Args:
            num_documents: Number of documents processed.
            total_ms: Total operation time in milliseconds.

        Returns:
            Timing metrics object.
        """
        return type(
            "TimingMetrics",
            (),
            {
                "tenant_resolution_ms": 0.0,
                "index_operation_ms": total_ms,
                "retrieval_ms": 0.0,
                "total_ms": total_ms,
                "tenant_id": self.tenant_context.tenant_id,
                "num_documents": num_documents,
            },
        )()
