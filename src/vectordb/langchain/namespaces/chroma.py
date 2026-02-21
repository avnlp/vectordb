"""Chroma namespace pipeline using collection-per-namespace pattern.

This module implements the NamespacePipeline interface for Chroma,
using separate collections for namespace isolation.
"""

import logging

from langchain_core.documents import Document

from vectordb.databases.chroma import ChromaVectorDB

from .base import NamespacePipeline
from .types import (
    CrossNamespaceComparison,
    CrossNamespaceResult,
    IsolationStrategy,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    NamespaceTimingMetrics,
    TenantStatus,
)
from .utils import Timer


logger = logging.getLogger(__name__)


class ChromaNamespacePipeline(NamespacePipeline):
    """Chroma namespace pipeline using collection-per-namespace.

    Each namespace maps to a separate Chroma collection, providing
    complete isolation between namespaces.
    """

    ISOLATION_STRATEGY = IsolationStrategy.COLLECTION

    def __init__(
        self,
        path: str = "./chroma_data",
        collection_prefix: str = "ns_",
    ) -> None:
        """Initialize Chroma namespace pipeline.

        Args:
            path: Path to Chroma data directory.
            collection_prefix: Prefix for namespace collection names.
        """
        self.db = ChromaVectorDB(path=path)
        self.path = path
        self.collection_prefix = collection_prefix

        logger.info(
            "Initialized Chroma namespace pipeline at path: %s",
            path,
        )

    def _get_collection_name(self, namespace: str) -> str:
        """Generate collection name for a namespace.

        Args:
            namespace: Namespace identifier.

        Returns:
            Collection name with prefix.
        """
        return f"{self.collection_prefix}{namespace}"

    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create a collection for the namespace."""
        collection_name = self._get_collection_name(namespace)
        self.db.create_collection(collection_name=collection_name)
        logger.info("Created namespace collection: %s", collection_name)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="create",
            message=f"Created collection '{collection_name}'",
        )

    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete the collection for a namespace."""
        collection_name = self._get_collection_name(namespace)
        self.db.delete_collection(collection_name=collection_name)
        logger.info("Deleted namespace collection: %s", collection_name)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="delete",
            message=f"Deleted collection '{collection_name}'",
        )

    def list_namespaces(self) -> list[str]:
        """List all namespaces (collections with matching prefix)."""
        all_collections = self.db.list_collections()
        namespaces = []
        for coll in all_collections:
            if coll.startswith(self.collection_prefix):
                namespaces.append(coll[len(self.collection_prefix) :])
        return namespaces

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace collection exists."""
        collection_name = self._get_collection_name(namespace)
        return collection_name in self.db.list_collections()

    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Get statistics for a namespace."""
        collection_name = self._get_collection_name(namespace)
        count = self.db._get_collection(collection_name).count()

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
        """Index documents with pre-computed embeddings into a namespace collection."""
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

        collection_name = self._get_collection_name(namespace)
        count = self.db.upsert(
            documents=documents,
            embeddings=embeddings,
            collection_name=collection_name,
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
        """Query a specific namespace collection."""
        return []

    def query_cross_namespace(
        self,
        query: str,
        namespaces: list[str] | None = None,
        top_k: int = 10,
    ) -> CrossNamespaceResult:
        """Query across multiple namespaces with timing comparison."""
        if namespaces is None:
            namespaces = self.list_namespaces()

        namespace_results: dict[str, list[NamespaceQueryResult]] = {}
        timing_comparisons: list[CrossNamespaceComparison] = []

        with Timer() as total_timer:
            for ns in namespaces:
                results = self.query_namespace(query, ns, top_k)
                namespace_results[ns] = results

                timing = NamespaceTimingMetrics(
                    namespace_lookup_ms=0.0,
                    vector_search_ms=0.0,
                    total_ms=0.0,
                    documents_searched=0,
                    documents_returned=len(results),
                )
                comparison = CrossNamespaceComparison(
                    namespace=ns,
                    timing=timing,
                    result_count=len(results),
                    top_score=results[0].relevance_score if results else 0.0,
                )
                timing_comparisons.append(comparison)

        return CrossNamespaceResult(
            query=query,
            namespace_results=namespace_results,
            timing_comparison=timing_comparisons,
            total_time_ms=total_timer.elapsed_ms,
        )
