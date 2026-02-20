"""Chroma namespace pipeline using collection-per-namespace pattern."""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document

from vectordb.databases.chroma import ChromaVectorDB
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


class ChromaNamespacePipeline:
    """Chroma namespace pipeline using collection-per-namespace.

    Uses ChromaVectorDB wrapper for all database operations.
    Implements namespace isolation using separate collections per namespace.
    """

    ISOLATION_STRATEGY = IsolationStrategy.COLLECTION

    def __init__(self, config_path: str) -> None:
        """Initialize the pipeline.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        self.config_path = config_path

        # Lazy-initialized components
        self._db: ChromaVectorDB | None = None
        self._doc_embedder = None
        self._text_embedder = None
        self._logger: logging.Logger | None = None

        self._connect()

    def _connect(self) -> None:
        """Connect to Chroma via unified wrapper."""
        self._db = ChromaVectorDB(config=self.config)

        embedding_dim = self.config.get("embedding", {}).get("dimension", 1024)
        self._db.create_index(dimension=embedding_dim)

    @property
    def db(self) -> ChromaVectorDB:
        """Get the VectorDB wrapper."""
        if self._db is None:
            raise RuntimeError("Not connected to Chroma")
        return self._db

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        if self._logger is None:
            name = self.config.get("pipeline", {}).get("name", "chroma_namespaces")
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
        self.logger.info("Closed Chroma connection")

    def __enter__(self) -> "ChromaNamespacePipeline":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()

    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create a collection for the namespace."""
        collection_name = (
            f"{self.config.get('collection', {}).get('name', 'ns')}_{namespace}"
        )
        self.db.create_collection(collection_name=collection_name)
        self.logger.info("Created namespace collection: %s", collection_name)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="create",
            message=f"Created collection '{collection_name}'",
        )

    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete the collection for a namespace."""
        collection_name = (
            f"{self.config.get('collection', {}).get('name', 'ns')}_{namespace}"
        )
        self.db.delete_collection(collection_name=collection_name)
        self.logger.info("Deleted namespace collection: %s", collection_name)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="delete",
            message=f"Deleted collection '{collection_name}'",
        )

    def list_namespaces(self) -> list[str]:
        """List all namespaces (collections with matching prefix)."""
        all_collections = self.db.list_collections()
        prefix = self.config.get("collection", {}).get("name", "ns")
        namespaces = []
        for coll in all_collections:
            if coll.startswith(f"{prefix}_"):
                namespaces.append(coll[len(f"{prefix}_") :])
        return namespaces

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace collection exists."""
        collection_name = (
            f"{self.config.get('collection', {}).get('name', 'ns')}_{namespace}"
        )
        return collection_name in self.db.list_collections()

    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Get statistics for a namespace."""
        collection_name = (
            f"{self.config.get('collection', {}).get('name', 'ns')}_{namespace}"
        )

        results = self.db.query(
            vector=[0.0] * 1024, top_k=10000, collection_name=collection_name
        )  # dummy query
        count = len(results)

        return NamespaceStats(
            namespace=namespace,
            document_count=count,
            vector_count=count,
            status=TenantStatus.ACTIVE if count > 0 else TenantStatus.UNKNOWN,
        )

    def index_documents(
        self, documents: list[Document], namespace: str
    ) -> NamespaceOperationResult:
        """Index documents into a namespace collection."""
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

        self._init_embedders()

        for doc in documents:
            if doc.meta is None:
                doc.meta = {}
            doc.meta["namespace"] = namespace

        embedded_docs = self._doc_embedder.run(documents=documents)["documents"]

        collection_name = (
            f"{self.config.get('collection', {}).get('name', 'ns')}_{namespace}"
        )
        count = self.db.upsert(
            data=embedded_docs,
            collection_name=collection_name,
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
        """Query a specific namespace collection."""
        self._init_embedders()

        with Timer() as embed_timer:
            query_embedding = self._text_embedder.run(text=query)["embedding"]

        collection_name = (
            f"{self.config.get('collection', {}).get('name', 'ns')}_{namespace}"
        )
        with Timer() as search_timer:
            results = self.db.query(
                vector=query_embedding, top_k=top_k, collection_name=collection_name
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
