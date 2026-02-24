"""Milvus multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from haystack import Document

from vectordb.databases.milvus import MilvusVectorDB
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


class MilvusMultitenancyIndexingPipeline:
    """Milvus indexing pipeline with partition key tenant isolation.

    Milvus partition keys provide:
    - Automatic data routing based on partition key value
    - Requires expr filter on every query for isolation
    - Up to 1024 partitions per collection
    """

    TENANT_FIELD = "tenant_id"

    def __init__(
        self,
        config_or_path: dict[str, Any] | str | Path,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize Milvus indexing pipeline.

        Args:
            config_or_path: Config dict or path to YAML file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self.config = load_config(config_or_path)
        self.tenant_context = TenantContext.resolve(tenant_context, self.config)
        self._db: MilvusVectorDB | None = None
        self._embedder = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Milvus and initialize embedder."""
        milvus_config = self.config.get("milvus", self.config.get("database", {}))

        self._db = MilvusVectorDB(
            uri=milvus_config.get(
                "uri", os.environ.get("MILVUS_URI", "http://localhost:19530")
            ),
            token=milvus_config.get("token", os.environ.get("MILVUS_TOKEN", "")),
            collection_name=self._get_collection_name(),
        )

        embedding_dim = self.config.get("embedding", {}).get("dimension", 1024)
        self._db.create_collection(
            collection_name=self._get_collection_name(),
            dimension=embedding_dim,
            use_partition_key=True,
            partition_key_field=self.TENANT_FIELD,
        )

        self._embedder = create_document_embedder(self.config)
        self._embedder.warm_up()

    def _get_collection_name(self) -> str:
        """Get collection name from config."""
        return self.config.get("collection", {}).get("name", "multitenancy")

    def close(self) -> None:
        """Close Milvus connection."""
        if self._db:
            self._db.close()

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
                    collection_name=self._get_collection_name(),
                    timing=self._create_timing_metrics(0, timer.elapsed_ms),
                )

            # Embed documents
            result = self._embedder.run(documents=documents)
            embedded_docs = result["documents"]

            # Truncate if output_dimension specified
            output_dim = self.config.get("embedding", {}).get("output_dimension")
            embedded_docs = truncate_embeddings(embedded_docs, output_dim)

            for doc in embedded_docs:
                doc.meta[self.TENANT_FIELD] = target_tenant

            # Insert with partition key
            self._db.insert_documents(
                documents=embedded_docs,
                collection_name=self._get_collection_name(),
            )

        metrics = self._create_timing_metrics(len(embedded_docs), timer.elapsed_ms)

        result = TenantIndexResult(
            tenant_id=target_tenant,
            documents_indexed=len(embedded_docs),
            collection_name=self._get_collection_name(),
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
        # In a real implementation, you might want to use the types from common.types
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
