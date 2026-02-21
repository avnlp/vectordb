"""Milvus namespace pipeline using partition key field.

This module implements the NamespacePipeline interface for Milvus,
using metadata field filtering for namespace isolation.
"""

import logging

from langchain_core.documents import Document

from vectordb.databases.milvus import MilvusVectorDB

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


class MilvusNamespacePipeline(NamespacePipeline):
    """Milvus namespace pipeline using partition key field.

    Each namespace is implemented using metadata field filtering,
    providing logical isolation within a single collection.
    """

    ISOLATION_STRATEGY = IsolationStrategy.PARTITION_KEY

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "namespaces",
        dimension: int = 384,
    ) -> None:
        """Initialize Milvus namespace pipeline.

        Args:
            host: Milvus host address.
            port: Milvus port number.
            collection_name: Name of the Milvus collection.
            dimension: Embedding dimension (default: 384).
        """
        self.db = MilvusVectorDB(
            host=host,
            port=port,
        )
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension

        logger.info(
            "Initialized Milvus namespace pipeline at %s:%d",
            host,
            port,
        )

    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create a namespace (auto-created on first upsert in Milvus)."""
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="create",
            message="Partition key namespaces are auto-created on insert (Milvus behavior)",
        )

    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete all documents in a namespace."""
        self.db.delete(filters={"namespace": namespace})
        logger.info("Deleted namespace: %s", namespace)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="delete",
            message=f"Deleted all documents in namespace '{namespace}'",
        )

    def list_namespaces(self) -> list[str]:
        """List all namespace values in the collection."""
        results = self.db.client.query(
            collection_name=self.db.collection_name,
            filter="",
            output_fields=["namespace"],
            limit=10000,
        )
        return list({r["namespace"] for r in results if "namespace" in r})

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace has any documents."""
        escaped = self.db._escape_expr_string(namespace)
        results = self.db.client.query(
            collection_name=self.db.collection_name,
            filter=f'namespace == "{escaped}"',
            output_fields=["count(*)"],
        )
        return results[0]["count(*)"] > 0 if results else False

    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Get statistics for a namespace."""
        escaped = self.db._escape_expr_string(namespace)
        results = self.db.client.query(
            collection_name=self.db.collection_name,
            filter=f'namespace == "{escaped}"',
            output_fields=["count(*)"],
        )
        count = results[0]["count(*)"] if results else 0

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
            collection_name=self.collection_name,
            partition_name=namespace,
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
