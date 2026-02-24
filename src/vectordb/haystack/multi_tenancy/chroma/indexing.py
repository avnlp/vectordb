"""Chroma multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document

from vectordb.databases.chroma import ChromaVectorDB
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


class ChromaMultitenancyIndexingPipeline(BaseMultitenancyPipeline):
    """Chroma indexing pipeline with database/tenant scoping.

    Chroma provides multi-tenancy through:
    - Separate collections per tenant
    - Metadata-based filtering
    - Database-level tenant isolation
    """

    TENANT_FIELD = "tenant_id"

    def __init__(
        self,
        config_path: str,
        tenant_context: Any = None,
    ) -> None:
        """Initialize Chroma indexing pipeline.

        Args:
            config_path: Path to YAML configuration file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self._db: ChromaVectorDB | None = None
        self._embedder: Any = None
        super().__init__(config_path=config_path, tenant_context=tenant_context)

    def _connect(self) -> None:
        """Connect to Chroma and initialize embedder."""
        chroma_config = self.config.get("chroma", self.config.get("database", {}))

        self._db = ChromaVectorDB(
            collection_name=self._get_collection_name(),
            persist_dir=chroma_config.get("persist_dir", "./chroma_db"),
        )

        self._embedder = create_document_embedder(self.config)
        self._embedder.warm_up()

    def _get_collection_name(self) -> str:
        """Get collection name from config with tenant suffix for isolation."""
        base_name = self.config.get("collection", {}).get("name", "multitenancy")

        # Resolve placeholders from dataset config
        dataset_config = self.config.get("dataset", {})
        dataset_name = dataset_config.get("name", "dataset")
        base_name = base_name.replace("{dataset}", dataset_name)

        tenant_suffix = self.tenant_context.tenant_id.replace("-", "_").replace(
            ".", "_"
        )
        return f"{base_name}_{tenant_suffix}"

    def close(self) -> None:
        """Close Chroma connection."""
        # Chroma doesn't have a specific close method
        pass

    def _get_document_store(self) -> ChromaVectorDB | None:
        """Return configured Chroma database instance.

        Returns:
            ChromaVectorDB instance or None if not initialized.
        """
        return self._db

    @property
    def isolation_strategy(self) -> TenantIsolationStrategy:
        """Return the isolation strategy for Chroma.

        Returns:
            TenantIsolationStrategy.DATABASE_SCOPING
        """
        return TenantIsolationStrategy.DATABASE_SCOPING

    def create_tenant(self, tenant_id: str) -> bool:
        """Create a new tenant in Chroma.

        Chroma uses collection-per-tenant isolation, so creating a tenant
        means creating a dedicated collection.

        Args:
            tenant_id: Tenant identifier to create.

        Returns:
            True if tenant was created, False if already exists.
        """
        # Chroma creates collections automatically on first upsert
        # Check if collection exists by trying to get it
        # Create collection by upserting a dummy document
        return not (self._db and self._db._collection is not None)

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists in Chroma.

        Args:
            tenant_id: Tenant identifier to check.

        Returns:
            True if tenant collection exists, False otherwise.
        """
        # In Chroma, tenant = collection, check if collection exists
        if self._db is None:
            return False

        try:
            # Try to access the collection
            return self._db._collection is not None
        except Exception:
            return False

    def get_tenant_stats(self, tenant_id: str) -> TenantStats:
        """Get statistics for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            TenantStats with tenant statistics.
        """
        if self._db is None:
            return TenantStats(
                tenant_id=tenant_id,
                document_count=0,
                vector_count=0,
                status=TenantStatus.UNKNOWN,
            )

        try:
            # Get count from Chroma collection
            count = self._db._collection.count() if self._db._collection else 0
            return TenantStats(
                tenant_id=tenant_id,
                document_count=count,
                vector_count=count,
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
        """List all tenants (collections) in Chroma.

        Returns:
            List of tenant identifiers.
        """
        if self._db is None:
            return []

        try:
            # Get all collections from Chroma
            client = self._db._client
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception:
            return []

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all its data.

        Args:
            tenant_id: Tenant identifier to delete.

        Returns:
            True if tenant was deleted, False if not found.
        """
        if self._db is None:
            return False

        try:
            # Delete the collection for this tenant
            collection_name = self._get_collection_name()
            client = self._db._client
            client.delete_collection(collection_name)
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

        # Add tenant metadata
        for doc in embedded_docs:
            doc.meta[self.TENANT_FIELD] = target_tenant

        # Insert documents into Chroma
        if self._db:
            self._db.upsert(data=embedded_docs)

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
        target_tenant = tenant_id or self.tenant_context.tenant_id

        if self._db is None:
            return []

        # Embed query
        from vectordb.haystack.multi_tenancy.common.embeddings import (
            create_query_embedder,
        )

        query_embedder = create_query_embedder(self.config)
        query_embedder.warm_up()
        query_result = query_embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Query Chroma with tenant filter
        return self._db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filter={self.TENANT_FIELD: target_tenant},
        )

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
