"""Pinecone namespace pipeline using native namespace isolation.

This module implements the NamespacePipeline interface for Pinecone,
using Pinecone's namespace mechanism for data isolation.
"""

import logging

from langchain_core.documents import Document

from vectordb.databases.pinecone import PineconeVectorDB

from .base import NamespacePipeline
from .types import (
    IsolationStrategy,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    TenantStatus,
)


logger = logging.getLogger(__name__)


class PineconeNamespacePipeline(NamespacePipeline):
    """Pinecone namespace pipeline using native namespace isolation.

    Each namespace maps to a Pinecone namespace within a single index,
    providing strong isolation with efficient vector operations.
    """

    ISOLATION_STRATEGY = IsolationStrategy.NAMESPACE

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 384,
    ) -> None:
        """Initialize Pinecone namespace pipeline.

        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            dimension: Embedding dimension (default: 384).
        """
        self.db = PineconeVectorDB(
            api_key=api_key,
            index_name=index_name,
        )
        self.index_name = index_name
        self.dimension = dimension

        logger.info(
            "Initialized Pinecone namespace pipeline for index: %s",
            index_name,
        )

    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create a namespace (auto-created on first upsert in Pinecone)."""
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="create",
            message="Namespace will be auto-created on first upsert (Pinecone behavior)",
        )

    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete all vectors in a namespace."""
        self.db.delete(delete_all=True, namespace=namespace)
        logger.info("Deleted all vectors in namespace: %s", namespace)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="delete",
            message=f"Deleted all vectors in namespace '{namespace}'",
        )

    def list_namespaces(self) -> list[str]:
        """List all namespaces in the index."""
        return self.db.list_namespaces()

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists."""
        return namespace in self.list_namespaces()

    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Get statistics for a namespace."""
        stats = self.db.describe_index_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        vector_count = ns_stats.get("vector_count", 0)

        return NamespaceStats(
            namespace=namespace,
            document_count=vector_count,
            vector_count=vector_count,
            status=TenantStatus.ACTIVE if vector_count > 0 else TenantStatus.UNKNOWN,
        )

    def index_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        namespace: str,
    ) -> NamespaceOperationResult:
        """Index documents with pre-computed embeddings into a namespace."""
        if not documents:
            return NamespaceOperationResult(
                success=True,
                namespace=namespace,
                operation="index",
                message="No documents to index",
                data={"count": 0},
            )

        count = self.db.upsert(
            documents=documents,
            embeddings=embeddings,
            namespace=namespace,
        )

        logger.info("Indexed %d documents into namespace '%s'", count, namespace)
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
        """Query a specific namespace."""
        return []
