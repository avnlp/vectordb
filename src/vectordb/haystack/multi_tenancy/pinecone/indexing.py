"""Pinecone multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

from haystack import Document
from pinecone.exceptions import PineconeException

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


class PineconeMultitenancyIndexingPipeline(BaseMultitenancyPipeline):
    """Pinecone indexing pipeline with namespace tenant isolation.

    Pinecone provides multi-tenancy through:
    - One namespace per tenant
    - Namespace-based isolation
    - Automatic tenant separation
    """

    def __init__(
        self,
        config_path: str,
        tenant_context: Any = None,
    ) -> None:
        """Initialize Pinecone indexing pipeline.

        Args:
            config_path: Path to YAML configuration file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self._pc: Any = None
        self._index: Any = None
        self._embedder: Any = None
        super().__init__(config_path=config_path, tenant_context=tenant_context)

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
            dimension = self._get_embedding_dimension()
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
        # Pinecone doesn't have a specific close method
        pass

    def _get_document_store(self) -> Any:
        """Return configured Pinecone index.

        Returns:
            Pinecone Index instance or None if not initialized.
        """
        return self._index

    @property
    def isolation_strategy(self) -> TenantIsolationStrategy:
        """Return the isolation strategy for Pinecone.

        Returns:
            TenantIsolationStrategy.NAMESPACE
        """
        return TenantIsolationStrategy.NAMESPACE

    def create_tenant(self, tenant_id: str) -> bool:
        """Create a new tenant in Pinecone.

        Pinecone uses namespaces, which are created implicitly on first use.

        Args:
            tenant_id: Tenant identifier to create.

        Returns:
            True if tenant was created, False if already exists.
        """
        # Namespaces are implicit in Pinecone - they exist once vectors are upserted
        return self.tenant_exists(tenant_id)

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists in Pinecone.

        Args:
            tenant_id: Tenant identifier to check.

        Returns:
            True if tenant namespace has vectors, False otherwise.
        """
        if self._index is None:
            return False

        try:
            # Check if namespace has any vectors
            stats = self._index.describe_index_stats(filter={"tenant_id": tenant_id})
            return stats.get("total_vector_count", 0) > 0
        except PineconeException:
            return False

    def get_tenant_stats(self, tenant_id: str) -> TenantStats:
        """Get statistics for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            TenantStats with tenant statistics.
        """
        if self._index is None:
            return TenantStats(
                tenant_id=tenant_id,
                document_count=0,
                vector_count=0,
                status=TenantStatus.UNKNOWN,
            )

        try:
            stats = self._index.describe_index_stats(filter={"tenant_id": tenant_id})
            count = stats.get("total_vector_count", 0)
            return TenantStats(
                tenant_id=tenant_id,
                document_count=count,
                vector_count=count,
                status=TenantStatus.ACTIVE,
            )
        except PineconeException:
            return TenantStats(
                tenant_id=tenant_id,
                document_count=0,
                vector_count=0,
                status=TenantStatus.UNKNOWN,
            )

    def list_tenants(self) -> list[str]:
        """List all tenants in Pinecone.

        Returns:
            List of tenant identifiers.
        """
        if self._index is None:
            return []

        try:
            # Get index stats to see all namespaces
            stats = self._index.describe_index_stats()
            namespaces = stats.get("namespaces", {})
            return sorted(namespaces.keys())
        except PineconeException:
            return []

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all its data.

        Args:
            tenant_id: Tenant identifier to delete.

        Returns:
            True if tenant was deleted, False if not found.
        """
        if self._index is None:
            return False

        try:
            # Delete all vectors in the tenant namespace
            self._index.delete(delete_all=True, namespace=tenant_id)
            return True
        except PineconeException:
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
                    namespace=target_tenant,
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

        if self._index is None:
            return []

        # Embed query
        from vectordb.haystack.multi_tenancy.common.embeddings import (
            create_query_embedder,
        )

        query_embedder = create_query_embedder(self.config)
        query_embedder.warm_up()
        query_result = query_embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Query Pinecone with tenant namespace
        results = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=True,
            namespace=target_tenant,
        )

        # Convert to Haystack Documents
        haystack_docs = []
        for match in results.matches:
            doc = Document(
                content=match.metadata.get("content", ""),
                meta={k: v for k, v in match.metadata.items() if k != "content"},
                embedding=match.values if hasattr(match, "values") else None,
                score=match.score,
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
                    collection_name=self._get_index_name(),
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
            collection_name=self._get_index_name(),
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
