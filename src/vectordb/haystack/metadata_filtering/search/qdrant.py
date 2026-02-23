"""Qdrant metadata filtering search pipeline for Haystack.

Performs vector search with metadata filtering on Qdrant.
"""

import logging

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.metadata_filtering.common import (
    FilteredQueryResult,
    Timer,
    TimingMetrics,
    create_rag_generator,
    filter_spec_to_canonical_dict,
    generate_answer,
    get_text_embedder,
    load_metadata_filtering_config,
    parse_filter_from_config,
)


__all__ = ["QdrantMetadataFilteringSearchPipeline"]

logger = logging.getLogger(__name__)


class QdrantMetadataFilteringSearchPipeline:
    """Qdrant metadata filtering search pipeline.

    Searches indexed documents with metadata filtering and optional RAG.

    Attributes:
        config: Configuration dictionary.
        db: QdrantVectorDB instance.
    """

    def __init__(self, config_or_path: str | dict) -> None:
        """Initialize Qdrant search pipeline from configuration.

        Args:
            config_or_path: Path to YAML config file or dict.

        Raises:
            ValueError: If config is invalid or required fields missing.
        """
        self.config = load_metadata_filtering_config(config_or_path)
        self._validate_config()
        self.db = self._init_db()
        logger.info("Initialized Qdrant search pipeline")

    def _validate_config(self) -> None:
        """Validate that all required config sections exist."""
        required_sections = ["embeddings", "qdrant", "search"]
        for section in required_sections:
            if section not in self.config or not self.config[section]:
                raise ValueError(f"Missing or empty '{section}' in configuration")

    def _init_db(self) -> QdrantVectorDB:
        """Initialize Qdrant connection.

        Returns:
            Initialized QdrantVectorDB instance.
        """
        qdrant_config = self.config["qdrant"]
        return QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key", ""),
            collection_name=qdrant_config.get("collection_name"),
        )

    def search(self, query: str | None = None) -> list[FilteredQueryResult]:
        """Execute filtered search with optional RAG.

        Args:
            query: Optional query text. Uses test_query from config if not
                provided.

        Returns:
            List of FilteredQueryResult with timing metrics.

        Raises:
            ValueError: If search fails.
        """
        metadata_filtering = self.config.get("metadata_filtering", {})
        query = query or metadata_filtering.get("test_query", "test query")
        top_k = self.config.get("search", {}).get("top_k", 10)

        logger.info("Searching with query: %s (top_k=%d)", query, top_k)

        # 1. Embed query
        logger.info("Embedding query...")
        text_embedder = get_text_embedder(self.config)
        query_result = text_embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # 2. Parse filter
        filter_spec = parse_filter_from_config(self.config)
        filter_dict = filter_spec_to_canonical_dict(filter_spec)
        logger.info("Filter: %s", filter_dict)

        # 3. Search with filter
        logger.info("Searching Qdrant...")
        with Timer() as search_timer:
            search_results = self.db.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filter_dict if filter_dict else None,
            )

        # 4. Build results
        timing = TimingMetrics(
            pre_filter_ms=0.0,
            vector_search_ms=search_timer.elapsed_ms,
            total_ms=search_timer.elapsed_ms,
            num_candidates=-1,
            num_total_docs=-1,
        )

        results = []
        for rank, doc in enumerate(search_results, start=1):
            results.append(
                FilteredQueryResult(
                    document=doc,
                    relevance_score=doc.score or 0.0,
                    rank=rank,
                    filter_matched=True,
                    timing=timing if rank == 1 else None,
                )
            )

        logger.info("Retrieved %d results", len(results))

        # 5. Optional RAG
        generator = create_rag_generator(self.config)
        if generator and results:
            docs = [r.document for r in results]
            answer = generate_answer(query, docs, generator)
            if answer:
                logger.info("Generated RAG answer: %s...", answer[:100])

        return results
