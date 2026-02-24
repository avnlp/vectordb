"""Qdrant multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.common.embeddings import (
    create_document_embedder,
    truncate_embeddings,
)
from vectordb.haystack.multi_tenancy.common.types import (
    MultitenancyTimingMetrics,
    TenantIndexResult,
    TenantIsolationStrategy,
    TenantStats,
    TenantStatus,
)


logger = logging.getLogger(__name__)


class QdrantMultitenancyIndexingPipeline(BaseMultitenancyPipeline):
    """Qdrant indexing pipeline with tenant payload isolation.

    Qdrant provides multi-tenancy through:
    - Payload-based filtering for tenant isolation
    - Collection-level tenant scoping
    - Efficient filtering with payload indices
    """

    TENANT_FIELD = "tenant_id"

    def __init__(
        self,
        config_path: str,
        tenant_context: Any = None,
    ) -> None:
        """Initialize Qdrant indexing pipeline.

        Args:
            config_path: Path to YAML configuration file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self._client: Any = None
        self._embedder: Any = None
        super().__init__(config_path=config_path, tenant_context=tenant_context)

    def _connect(self) -> None:
        """Connect to Qdrant and initialize embedder."""
        import os

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
        embedding_dim = self._get_embedding_dimension()

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

    def _get_document_store(self) -> Any:
        """Return configured Qdrant client.

        Returns:
            QdrantClient instance or None if not initialized.
        """
        return self._client

    @property
    def isolation_strategy(self) -> TenantIsolationStrategy:
        """Return the isolation strategy for Qdrant.

        Returns:
            TenantIsolationStrategy.TIERED
        """
        return TenantIsolationStrategy.TIERED

    def create_tenant(self, tenant_id: str) -> bool:
        """Create a new tenant in Qdrant.

        Qdrant uses payload-based filtering, so tenants are created
        implicitly on first document insertion.

        Args:
            tenant_id: Tenant identifier to create.

        Returns:
            True if tenant was created, False if already exists.
        """
        # Tenants are implicit in Qdrant - they exist once documents are indexed
        return self.tenant_exists(tenant_id)

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists in Qdrant.

        Args:
            tenant_id: Tenant identifier to check.

        Returns:
            True if tenant has documents, False otherwise.
        """
        if self._client is None:
            return False

        try:
            # Check if any documents exist with this tenant_id
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue

            result = self._client.count(
                collection_name=self._get_collection_name(),
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key=self.TENANT_FIELD,
                            match=MatchValue(value=tenant_id),
                        )
                    ]
                ),
            )
            return result.count > 0
        except Exception:
            return False

    def get_tenant_stats(self, tenant_id: str) -> TenantStats:
        """Get statistics for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            TenantStats with tenant statistics.
        """
        if self._client is None:
            return TenantStats(
                tenant_id=tenant_id,
                document_count=0,
                vector_count=0,
                status=TenantStatus.UNKNOWN,
            )

        try:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue

            result = self._client.count(
                collection_name=self._get_collection_name(),
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key=self.TENANT_FIELD,
                            match=MatchValue(value=tenant_id),
                        )
                    ]
                ),
            )
            return TenantStats(
                tenant_id=tenant_id,
                document_count=result.count,
                vector_count=result.count,
                status=TenantStatus.ACTIVE,
            )
        except Exception:
            return TenantStats(
                tenant_id=tenant_id,
                document_count=0,
                vector_count=0,
                status=TenantStatus.UNKNOWN,
            )

    def list_tenants(self) -> list[str]:
        """List all tenants in Qdrant.

        Returns:
            List of tenant identifiers.
        """
        if self._client is None:
            return []

        try:
            # Scroll through all points to get unique tenant_ids
            all_tenants: set[str] = set()
            offset = None

            while True:
                records, offset = self._client.scroll(
                    collection_name=self._get_collection_name(),
                    limit=100,
                    with_payload=[self.TENANT_FIELD],
                    with_vectors=False,
                    offset=offset,
                )

                for record in records:
                    if record.payload and self.TENANT_FIELD in record.payload:
                        all_tenants.add(record.payload[self.TENANT_FIELD])

                if offset is None:
                    break

            return sorted(all_tenants)
        except Exception:
            return []

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all its data.

        Args:
            tenant_id: Tenant identifier to delete.

        Returns:
            True if tenant was deleted, False if not found.
        """
        if self._client is None:
            return False

        try:
            from qdrant_client.http.models import FieldCondition, Filter, MatchValue

            # Delete all points with this tenant_id
            self._client.delete(
                collection_name=self._get_collection_name(),
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key=self.TENANT_FIELD,
                            match=MatchValue(value=tenant_id),
                        )
                    ]
                ),
            )
            return True
        except Exception:
            return False

    def index_documents(
        self,
        documents: list[Document],
        tenant_id: str | None = None,
    ) -> int:
        """Index documents for a tenant.

        Args:
            documents: List of Haystack Documents to index.
            tenant_id: Target tenant. Uses current context if None.

        Returns:
            Number of documents indexed.
        """
        from qdrant_client.http.models import PointStruct

        target_tenant = tenant_id or self.tenant_context.tenant_id

        if not documents:
            self.logger.warning("No documents to index")
            return 0

        # Embed documents
        result = self._embedder.run(documents=documents)
        embedded_docs = result["documents"]

        # Truncate if output_dimension specified
        output_dim = self.config.get("embedding", {}).get("output_dimension")
        embedded_docs = truncate_embeddings(embedded_docs, output_dim)

        # Prepare points for Qdrant with tenant payload
        points = []
        for i, doc in enumerate(embedded_docs):
            if doc.embedding is not None:
                doc_id = f"{target_tenant}_{i}"

                # Prepare payload with tenant information
                payload = doc.meta.copy()
                payload[self.TENANT_FIELD] = target_tenant
                payload["content"] = doc.content or ""

                point = PointStruct(id=doc_id, vector=doc.embedding, payload=payload)
                points.append(point)

        # Upload points to Qdrant
        if points:
            self._client.upsert(
                collection_name=self._get_collection_name(), points=points
            )

        self.logger.info(
            "Indexed %d documents for tenant %s",
            len(embedded_docs),
            target_tenant,
        )

        return len(embedded_docs)

    def query(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
    ) -> list[Document]:
        """Query documents within tenant scope.

        Args:
            query: Query string.
            top_k: Number of results to return.
            tenant_id: Target tenant. Uses current context if None.

        Returns:
            List of retrieved Documents.
        """
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        target_tenant = tenant_id or self.tenant_context.tenant_id

        if self._client is None:
            return []

        # Embed query
        from vectordb.haystack.multi_tenancy.common.embeddings import (
            create_query_embedder,
        )

        query_embedder = create_query_embedder(self.config)
        query_embedder.warm_up()
        query_result = query_embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Query Qdrant with tenant filter
        results = self._client.query_points(
            collection_name=self._get_collection_name(),
            query=query_embedding,
            limit=top_k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key=self.TENANT_FIELD,
                        match=MatchValue(value=target_tenant),
                    )
                ]
            ),
        )

        # Convert to Haystack Documents
        haystack_docs = []
        for point in results.points:
            doc = Document(
                content=point.payload.get("content", ""),
                meta={k: v for k, v in point.payload.items() if k != "content"},
                embedding=point.vector if hasattr(point, "vector") else None,
            )
            haystack_docs.append(doc)

        return haystack_docs

    def run(
        self,
        documents: list[Document] | None = None,
        tenant_id: str | None = None,
    ) -> TenantIndexResult:
        """Index documents for a tenant (convenience method).

        This method provides a convenient way to index documents while
        capturing timing metrics. For simple indexing without metrics,
        use index_documents() directly.

        Args:
            documents: Documents to index. If None, loads from dataloader.
            tenant_id: Target tenant. Uses context if None.

        Returns:
            TenantIndexResult with indexing metrics.
        """
        from vectordb.haystack.multi_tenancy.base import Timer

        target_tenant = tenant_id or self.tenant_context.tenant_id

        with Timer() as timer:
            if documents is None:
                documents = self._load_documents_from_dataloader()

            if not documents:
                self.logger.warning("No documents to index")
                return TenantIndexResult(
                    tenant_id=target_tenant,
                    documents_indexed=0,
                    collection_name=self._get_collection_name(),
                    timing=MultitenancyTimingMetrics(
                        tenant_resolution_ms=0.0,
                        index_operation_ms=timer.elapsed_ms,
                        retrieval_ms=0.0,
                        total_ms=timer.elapsed_ms,
                        tenant_id=target_tenant,
                        num_documents=0,
                    ),
                )

            num_indexed = self.index_documents(documents, tenant_id)

        return TenantIndexResult(
            tenant_id=target_tenant,
            documents_indexed=num_indexed,
            collection_name=self._get_collection_name(),
            timing=MultitenancyTimingMetrics(
                tenant_resolution_ms=0.0,
                index_operation_ms=timer.elapsed_ms,
                retrieval_ms=0.0,
                total_ms=timer.elapsed_ms,
                tenant_id=target_tenant,
                num_documents=num_indexed,
            ),
        )

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
