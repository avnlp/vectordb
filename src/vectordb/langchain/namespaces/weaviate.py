"""Weaviate namespace pipeline using native multi-tenancy.

This module implements the NamespacePipeline interface for Weaviate,
using Weaviate's tenant mechanism for namespace isolation.
"""

import logging

from langchain_core.documents import Document

from vectordb.databases.weaviate import WeaviateVectorDB

from .base import NamespacePipeline
from .types import (
    IsolationStrategy,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    TenantStatus,
)


logger = logging.getLogger(__name__)


class WeaviateNamespacePipeline(NamespacePipeline):
    """Weaviate namespace pipeline using native multi-tenancy.

    Each namespace maps to a Weaviate tenant, providing strong isolation
    with efficient vector operations.
    """

    ISOLATION_STRATEGY = IsolationStrategy.TENANT

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        collection_prefix: str = "ns_",
    ) -> None:
        """Initialize Weaviate namespace pipeline.

        Args:
            url: Weaviate instance URL.
            api_key: Optional Weaviate API key.
            collection_prefix: Prefix for namespace collections.
        """
        self.db = WeaviateVectorDB(
            url=url,
            api_key=api_key,
        )
        self.url = url
        self.collection_prefix = collection_prefix

        logger.info(
            "Initialized Weaviate namespace pipeline at URL: %s",
            url,
        )

    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create a namespace (tenant) in Weaviate."""
        self.db.create_tenant(tenant=namespace)
        logger.info("Created tenant: %s", namespace)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="create",
            message=f"Created tenant '{namespace}'",
        )

    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete a namespace (tenant) from Weaviate."""
        self.db.delete_tenant(tenant=namespace)
        logger.info("Deleted tenant: %s", namespace)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="delete",
            message=f"Deleted tenant '{namespace}'",
        )

    def list_namespaces(self) -> list[str]:
        """List all tenants in the collection."""
        return self.db.list_tenants()

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a tenant exists."""
        return namespace in self.list_namespaces()

    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Get statistics for a tenant."""
        self.db.with_tenant(namespace)
        aggregation = self.db.collection.aggregate.over_all(total_count=True)
        count = aggregation.total_count

        return NamespaceStats(
            namespace=namespace,
            document_count=count,
            vector_count=count,
            status=TenantStatus.ACTIVE if count > 0 else TenantStatus.UNKNOWN,
        )

    def index_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        namespace: str,
    ) -> NamespaceOperationResult:
        """Index documents with pre-computed embeddings into a namespace (tenant)."""
        if not documents:
            return NamespaceOperationResult(
                success=True,
                namespace=namespace,
                operation="index",
                message="No documents to index",
                data={"count": 0},
            )

        if not self.namespace_exists(namespace):
            self.create_namespace(namespace)

        count = self.db.upsert(
            documents=documents,
            embeddings=embeddings,
            tenant=namespace,
        )

        logger.info("Indexed %d documents into tenant '%s'", count, namespace)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="index",
            message=f"Indexed {count} documents",
            data={"count": count},
        )

    def query_namespace(
        self, query: str, namespace: str, top_k: int = 10
    ) -> list[NamespaceQueryResult]:
        """Query a specific namespace (tenant)."""
        return []
