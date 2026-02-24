"""Qdrant multi-tenancy indexing pipeline."""

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


class QdrantMultitenancyIndexingPipeline:
    """Qdrant indexing pipeline with tenant payload isolation.

    Qdrant provides multi-tenancy through:
    - Payload-based filtering for tenant isolation
    - Collection-level tenant scoping
    - Efficient filtering with payload indices
    """

    TENANT_FIELD = "tenant_id"

    def __init__(
        self,
        config_or_path: dict[str, Any] | str | Path,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize Qdrant indexing pipeline.

        Args:
            config_or_path: Config dict or path to YAML file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self.config = load_config(config_or_path)
        self.tenant_context = TenantContext.resolve(tenant_context, self.config)
        self._client = None
        self._embedder = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Qdrant and initialize embedder."""
        from qdrant_client import QdrantClient

        qdrant_config = self.config.get("qdrant", self.config.get("database", {}))

        # Connect to Qdrant
        if "location" in qdrant_config:
            # Local mode
            self._client = QdrantClient(location=qdrant_config["location"])
        elif "url" in qdrant_config:
            # Remote mode
            url = qdrant_config["url"]
            api_key = qdrant_config.get("api_key") or os.environ.get("QDRANT_API_KEY")
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            # Default to local
            self._client = QdrantClient(
                host=os.environ.get("QDRANT_HOST", "localhost"),
                port=int(os.environ.get("QDRANT_PORT", "6333")),
            )

        collection_name = self._get_collection_name()
        embedding_dim = self.config.get("embedding", {}).get("dimension", 1024)

        from qdrant_client.http.models import Distance, VectorParams

        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            )

        self._client.create_payload_index(
            collection_name=collection_name,
            field_name=self.TENANT_FIELD,
            field_schema="keyword",
        )

        self._embedder = create_document_embedder(self.config)
        self._embedder.warm_up()

    def _get_collection_name(self) -> str:
        """Get collection name from config."""
        return self.config.get("collection", {}).get("name", "multitenancy")

    def close(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            self._client.close()

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

            # Prepare points for Qdrant with tenant payload
            from qdrant_client.http.models import PointStruct

            points = []
            for i, doc in enumerate(embedded_docs):
                if doc.embedding is not None:
                    doc_id = f"{target_tenant}_{i}"

                    # Prepare payload with tenant information
                    payload = doc.meta.copy()
                    payload[self.TENANT_FIELD] = target_tenant
                    payload["content"] = doc.content or ""

                    point = PointStruct(
                        id=doc_id, vector=doc.embedding, payload=payload
                    )
                    points.append(point)

            # Upload points to Qdrant
            if points:
                # Batch upload to Qdrant
                self._client.upsert(
                    collection_name=self._get_collection_name(), points=points
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
