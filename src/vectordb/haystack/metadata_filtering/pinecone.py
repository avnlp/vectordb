"""Pinecone metadata filtering pipeline for Haystack.

Refactored to use the unified PineconeVectorDB class.
"""

from typing import Any

from haystack import Document

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.metadata_filtering.base import (
    BaseMetadataFilteringPipeline,
    Timer,
)
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
    FilterSpec,
    PineconeFilterExpressionBuilder,
    TimingMetrics,
    parse_filter_from_config,
)


__all__ = ["PineconeMetadataFilteringPipeline"]


class PineconeMetadataFilteringPipeline(BaseMetadataFilteringPipeline):
    """Pinecone metadata filtering pipeline with pre-filter and vector search.

    Uses unified PineconeVectorDB for all operations.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize Pinecone pipeline from configuration."""
        self.vector_db: PineconeVectorDB | None = None
        self.namespace: str = "default"
        self.dimension: int = 384
        super().__init__(config_path)

    def _connect(self) -> None:
        """Establish connection via PineconeVectorDB."""
        self.vector_db = PineconeVectorDB(config=self.config)
        self.namespace = self.config.get("pinecone", {}).get("namespace", "default")
        self.logger.info("Connected to Pinecone via unified PineconeVectorDB")

    def _create_index(self, index_name: str, dimension: int) -> None:
        """Create or get index."""
        if not self.vector_db:
            raise ValueError("VectorDB not initialized")

        self.vector_db.create_index(
            index_name=index_name,
            dimension=dimension,
            metric="cosine",
        )
        self.dimension = dimension

    def _index_documents(self, documents: list[Document]) -> int:
        """Index documents using unified upsert."""
        if not self.vector_db:
            raise ValueError("VectorDB not initialized")

        return self.vector_db.upsert(
            data=documents,
            namespace=self.namespace,
        )

    def _pre_filter(self, filter_dict: dict[str, Any]) -> int:
        """Estimate matching candidates."""
        if not self.vector_db:
            return 0
        return self.vector_db.estimate_match_count(
            filter_dict, namespace=self.namespace
        )

    def _vector_search(
        self,
        query_embedding: list[float],
        filter_dict: dict[str, Any],
        top_k: int = 10,
    ) -> list[Document]:
        """Execute vector search with filter."""
        if not self.vector_db:
            raise ValueError("VectorDB not initialized")

        return self.vector_db.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            namespace=self.namespace,
            include_metadata=True,
        )

    def run(self) -> list[FilteredQueryResult]:
        """Execute complete metadata filtering pipeline."""
        self._init_embedder()

        pinecone_config = self.config.get("pinecone", {})
        index_name = pinecone_config.get("index_name", "metadata-filtered")
        embeddings_config = self.config.get("embeddings", {})
        self.dimension = embeddings_config.get("dimension", 384)

        self._create_index(index_name, self.dimension)

        # Placeholder for actual data loading

        filter_spec: FilterSpec = parse_filter_from_config(self.config)
        schema = self._get_metadata_schema()
        builder = PineconeFilterExpressionBuilder(schema)
        filter_dict = builder.build(filter_spec)

        test_query = self.config.get("metadata_filtering", {}).get(
            "test_query", "test query"
        )

        if self.embedder:
            from haystack.components.embedders import SentenceTransformersTextEmbedder

            model_name = self.config.get("embeddings", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            text_embedder = SentenceTransformersTextEmbedder(model=model_name)
            text_embedder.warm_up()
            query_embedding = text_embedder.run(text=test_query)["embedding"]
        else:
            query_embedding = [0.0] * self.dimension

        with Timer() as pre_filter_timer:
            num_candidates = self._pre_filter(filter_dict)

        with Timer() as search_timer:
            search_results = self._vector_search(query_embedding, filter_dict)

        stats = self.vector_db.describe_index_stats()
        total_docs = stats.get("total_vector_count", 0)

        results: list[FilteredQueryResult] = []
        timing = TimingMetrics(
            pre_filter_ms=pre_filter_timer.elapsed_ms,
            vector_search_ms=search_timer.elapsed_ms,
            total_ms=pre_filter_timer.elapsed_ms + search_timer.elapsed_ms,
            num_candidates=num_candidates,
            num_total_docs=total_docs,
        )

        for rank, doc in enumerate(search_results, start=1):
            results.append(
                FilteredQueryResult(
                    document=doc,
                    relevance_score=doc.score if hasattr(doc, "score") else 0.0,
                    rank=rank,
                    filter_matched=True,
                    timing=timing if rank == 1 else None,
                )
            )

        return results
