"""Milvus namespace pipeline using partition key field."""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.namespaces.types import (
    CrossNamespaceComparison,
    CrossNamespaceResult,
    IsolationStrategy,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    NamespaceTimingMetrics,
    TenantStatus,
)
from vectordb.haystack.namespaces.utils import (
    Timer,
    get_document_embedder,
    get_text_embedder,
    load_config,
    load_documents_from_config,
)


class MilvusNamespacePipeline:
    """Milvus namespace pipeline using partition key field.

    Uses MilvusVectorDB wrapper for all database operations.
    Implements namespace isolation using partition key fields.
    """

    ISOLATION_STRATEGY = IsolationStrategy.PARTITION_KEY

    def __init__(self, config_path: str) -> None:
        """Initialize the pipeline.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        self.config_path = config_path

        # Lazy-initialized components
        self._db: MilvusVectorDB | None = None
        self._doc_embedder = None
        self._text_embedder = None
        self._logger: logging.Logger | None = None

        self._connect()

    def _connect(self) -> None:
        """Connect to Milvus via unified wrapper."""
        self._db = MilvusVectorDB(config=self.config)

        embedding_dim = self.config.get("embedding", {}).get("dimension", 1024)
        self._db.create_index(dimension=embedding_dim)

    @property
    def db(self) -> MilvusVectorDB:
        """Get the VectorDB wrapper."""
        if self._db is None:
            raise RuntimeError("Not connected to Milvus")
        return self._db

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        if self._logger is None:
            name = self.config.get("pipeline", {}).get("name", "milvus_namespaces")
            self._logger = logging.getLogger(name)
        return self._logger

    def _init_embedders(self) -> None:
        """Initialize embedders lazily."""
        if self._doc_embedder is None:
            self._doc_embedder = get_document_embedder(self.config)
            self._text_embedder = get_text_embedder(self.config)

    def close(self) -> None:
        """Close connection."""
        self._db = None
        self.logger.info("Closed Milvus connection")

    def __enter__(self) -> "MilvusNamespacePipeline":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()

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
        # Use metadata filtering to delete documents in specific namespace
        self.db.delete(filters={"namespace": namespace})
        self.logger.info("Deleted namespace: %s", namespace)
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
        self, documents: list[Document], namespace: str
    ) -> NamespaceOperationResult:
        """Index documents into a namespace."""
        if not documents:
            return NamespaceOperationResult(
                success=True,
                namespace=namespace,
                operation="index",
                message="No documents to index",
                data={"count": 0},
            )

        self._init_embedders()

        for doc in documents:
            if doc.meta is None:
                doc.meta = {}
            doc.meta["namespace"] = namespace

        embedded_docs = self._doc_embedder.run(documents=documents)["documents"]

        count = self.db.upsert(
            data=embedded_docs,
            batch_size=self.config.get("indexing", {}).get("batch_size", 100),
        )

        self.logger.info("Indexed %d documents into namespace '%s'", count, namespace)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="index",
            message=f"Indexed {count} documents",
            data={"count": count},
        )

    def index_from_config(self, namespace: str) -> NamespaceOperationResult:
        """Index documents from dataset specified in config."""
        documents = load_documents_from_config(self.config)
        return self.index_documents(documents, namespace)

    def query_namespace(
        self, query: str, namespace: str, top_k: int = 10
    ) -> list[NamespaceQueryResult]:
        """Query a specific namespace."""
        self._init_embedders()

        with Timer() as embed_timer:
            query_embedding = self._text_embedder.run(text=query)["embedding"]

        with Timer() as search_timer:
            # Apply namespace filter in the query
            results = self.db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters={"namespace": namespace},
            )

        stats = self.get_namespace_stats(namespace)
        timing = NamespaceTimingMetrics(
            namespace_lookup_ms=embed_timer.elapsed_ms,
            vector_search_ms=search_timer.elapsed_ms,
            total_ms=embed_timer.elapsed_ms + search_timer.elapsed_ms,
            documents_searched=stats.document_count,
            documents_returned=len(results),
        )

        query_results = []
        for i, doc in enumerate(results):
            query_results.append(
                NamespaceQueryResult(
                    document=doc,
                    relevance_score=getattr(doc, "score", 0.0),
                    rank=i + 1,
                    namespace=namespace,
                    timing=timing if i == 0 else None,
                )
            )

        return query_results

    def query_cross_namespace(
        self, query: str, namespaces: list[str] | None = None, top_k: int = 10
    ) -> CrossNamespaceResult:
        """Query across multiple namespaces with timing comparison.

        Args:
            query: Query string.
            namespaces: List of namespaces to query. If None, queries all.
            top_k: Number of results per namespace.

        Returns:
            CrossNamespaceResult with results and timing comparison.
        """
        if namespaces is None:
            namespaces = self.list_namespaces()

        namespace_results: dict[str, list[NamespaceQueryResult]] = {}
        timing_comparisons: list[CrossNamespaceComparison] = []

        with Timer() as total_timer:
            for ns in namespaces:
                results = self.query_namespace(query, ns, top_k)
                namespace_results[ns] = results

                if results:
                    timing = results[0].timing
                    if timing is not None:
                        comparison = CrossNamespaceComparison(
                            namespace=ns,
                            timing=timing,
                            result_count=len(results),
                            top_score=results[0].relevance_score if results else 0.0,
                        )
                        timing_comparisons.append(comparison)
                    else:
                        comparison = CrossNamespaceComparison(
                            namespace=ns,
                            timing=NamespaceTimingMetrics(
                                namespace_lookup_ms=0.0,
                                vector_search_ms=0.0,
                                total_ms=0.0,
                                documents_searched=0,
                                documents_returned=len(results),
                            ),
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

    def run(self) -> dict[str, Any]:
        """Execute the complete namespace pipeline."""
        # This would implement the full pipeline execution
        # For now, return a placeholder result
        return {
            "success": True,
            "message": "Pipeline executed successfully",
            "namespaces": self.list_namespaces(),
        }
