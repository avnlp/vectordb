"""Milvus multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

from haystack import Document

from vectordb.databases.milvus import MilvusVectorDB
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


class MilvusMultitenancyIndexingPipeline(BaseMultitenancyPipeline):
    """Milvus indexing pipeline with partition key tenant isolation.

    Milvus partition keys provide:
    - Automatic data routing based on partition key value
    - Requires expr filter on every query for isolation
    - Up to 1024 partitions per collection
    """

    TENANT_FIELD = "tenant_id"

    def __init__(
        self,
        config_path: str,
        tenant_context: Any = None,
    ) -> None:
        """Initialize Milvus indexing pipeline.

        Args:
            config_path: Path to YAML configuration file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self._db: MilvusVectorDB | None = None
        self._embedder: Any = None
        super().__init__(config_path=config_path, tenant_context=tenant_context)

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

        embedding_dim = self._get_embedding_dimension()
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

    def _get_document_store(self) -> MilvusVectorDB | None:
        """Return configured Milvus database instance.

        Returns:
            MilvusVectorDB instance or None if not initialized.
        """
        return self._db

    @property
    def isolation_strategy(self) -> TenantIsolationStrategy:
        """Return the isolation strategy for Milvus.

        Returns:
            TenantIsolationStrategy.PARTITION_KEY
        """
        return TenantIsolationStrategy.PARTITION_KEY

    def create_tenant(self, tenant_id: str) -> bool:
        """Create a new tenant in Milvus.

        Milvus uses partition keys, so tenants are created implicitly
        on first document insertion.

        Args:
            tenant_id: Tenant identifier to create.

        Returns:
            True if tenant was created, False if already exists.
        """
        return self.tenant_exists(tenant_id)

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists in Milvus.

        Args:
            tenant_id: Tenant identifier to check.

        Returns:
            True if tenant has documents, False otherwise.
        """
        if self._db is None:
            return False

        try:
            # Query for any document with this tenant_id
            results = self._db.query(
                filter_expr=f"{self.TENANT_FIELD} == '{tenant_id}'",
                output_fields=[self.TENANT_FIELD],
                limit=1,
            )
            return len(results) > 0
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
            # Query for count of documents with this tenant_id
            results = self._db.query(
                filter_expr=f"{self.TENANT_FIELD} == '{tenant_id}'",
                output_fields=["count(*)"],
            )
            count = results[0].get("count(*)", 0) if results else 0
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
        """List all tenants in Milvus.

        Returns:
            List of tenant identifiers.
        """
        if self._db is None:
            return []

        try:
            # Query for distinct tenant_ids
            results = self._db.query(
                filter_expr="",
                output_fields=[self.TENANT_FIELD],
            )
            tenants = set()
            for result in results:
                if self.TENANT_FIELD in result:
                    tenants.add(result[self.TENANT_FIELD])
            return sorted(tenants)
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
            # Delete all documents with this tenant_id
            self._db.delete(
                filter_expr=f"{self.TENANT_FIELD} == '{tenant_id}'",
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

        # Insert with partition key
        self._db.insert_documents(
            documents=embedded_docs,
            collection_name=self._get_collection_name(),
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

        # Query Milvus with tenant filter
        return self._db.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_expr=f"{self.TENANT_FIELD} == '{target_tenant}'",
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
