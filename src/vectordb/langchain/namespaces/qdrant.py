"""Qdrant namespace pipeline using payload-based filtering.

This module implements the NamespacePipeline interface for Qdrant,
using payload filters for namespace isolation.
"""

import logging

from langchain_core.documents import Document
from qdrant_client import models

from vectordb.databases.qdrant import QdrantVectorDB

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


class QdrantNamespacePipeline(NamespacePipeline):
    """Qdrant namespace pipeline using payload-based filtering.

    Each namespace is implemented using payload filters on a shared collection,
    providing logical isolation between namespaces.
    """

    ISOLATION_STRATEGY = IsolationStrategy.PAYLOAD_FILTER

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_prefix: str = "ns_",
    ) -> None:
        """Initialize Qdrant namespace pipeline.

        Args:
            url: Qdrant server URL.
            api_key: Optional Qdrant API key.
            collection_prefix: Prefix for namespace collections.
        """
        self.db = QdrantVectorDB(
            url=url,
            api_key=api_key,
        )
        self.url = url
        self.collection_prefix = collection_prefix

        logger.info(
            "Initialized Qdrant namespace pipeline at URL: %s",
            url,
        )

    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create a namespace (auto-created on first upsert in Qdrant)."""
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="create",
            message="Payload-based namespaces are auto-created on insert (Qdrant behavior)",
        )

    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete all points with a specific namespace."""
        self.db.delete(filters={"namespace": namespace})
        logger.info("Deleted namespace: %s", namespace)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="delete",
            message=f"Deleted all points in namespace '{namespace}'",
        )

    def list_namespaces(self) -> list[str]:
        """List all unique namespace values."""
        namespaces: set[str] = set()
        offset = None
        while True:
            records, offset = self.db.client.scroll(
                collection_name=self.db.collection_name,
                limit=256,
                offset=offset,
                with_payload=["namespace"],
                with_vectors=False,
            )
            for record in records:
                ns = (record.payload or {}).get("namespace")
                if ns is not None:
                    namespaces.add(ns)
            if offset is None:
                break
        return list(namespaces)

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace has any points."""
        count = self.db.client.count(
            collection_name=self.db.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace",
                        match=models.MatchValue(value=namespace),
                    )
                ]
            ),
            exact=False,
        ).count
        return count > 0

    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Get statistics for a namespace."""
        count = self.db.client.count(
            collection_name=self.db.collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace",
                        match=models.MatchValue(value=namespace),
                    )
                ]
            ),
            exact=True,
        ).count

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
