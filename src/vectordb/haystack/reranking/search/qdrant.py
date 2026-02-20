"""Qdrant reranking search pipeline.

This module provides a search pipeline combining Qdrant's hybrid search
capabilities with cross-encoder reranking. Qdrant's support for both
dense and sparse vectors makes it ideal for hybrid retrieval scenarios.

Reranking Strategy:
    1. Embed query using bi-encoder for dense vector representation
    2. Retrieve 3x candidates from Qdrant using dense vector search
    3. Score candidates with cross-encoder for precise relevance
    4. Return top_k results ordered by cross-encoder score

Qdrant Advantages:
    - High-performance vector search with HNSW indexing
    - Payload filtering with rich query DSL
    - Distributed deployment options
    - Sparse vector support for BM25-style retrieval

Hybrid Search Integration:
    While this pipeline uses dense retrieval + cross-encoder reranking,
    Qdrant also supports hybrid dense-sparse search that could be used
    for the initial retrieval stage.
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory, RerankerFactory


logger = logging.getLogger(__name__)


class QdrantRerankingSearchPipeline:
    """Qdrant search pipeline with cross-encoder reranking.

    Combines Qdrant's efficient vector search with cross-encoder reranking
    for high-precision retrieval. The pipeline uses Qdrant's HNSW index
    for fast candidate retrieval, then applies accurate cross-encoder scoring.

    Attributes:
        config: Pipeline configuration with qdrant, embedder, reranker sections.
        embedder: Bi-encoder for query embedding generation.
        reranker: Cross-encoder for final document scoring.
        db: QdrantVectorDB instance for vector storage and search.
        collection_name: Name of the Qdrant collection to search.

    Example:
        >>> pipeline = QdrantRerankingSearchPipeline("qdrant_config.yaml")
        >>> results = pipeline.search(
        ...     query="deep learning architectures",
        ...     top_k=10,
        ...     filters={"category": "research_paper"},
        ... )
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - qdrant: url, collection_name, api_key (if cloud)
                - embedder: Provider, model, dimensions for bi-encoder
                - reranker: Provider, model for cross-encoder

        Raises:
            ValueError: If required config sections are missing or invalid.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        self.reranker = RerankerFactory.create(self.config)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            collection_name=qdrant_config.get("collection_name", "reranking"),
        )

        self.collection_name = qdrant_config.get("collection_name", "reranking")

        logger.info("Initialized Qdrant reranking search pipeline")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search with two-stage retrieval.

        Stage 1 - Dense Retrieval:
            Query is embedded using bi-encoder, then Qdrant's HNSW index
            retrieves 3x the requested top_k candidates using vector similarity.

        Stage 2 - Cross-Encoder Reranking:
            Cross-encoder scores each candidate for relevance to query.
            Only top_k highest-scoring documents are returned.

        Args:
            query: Search query text to find relevant documents for.
            top_k: Number of final results to return after reranking.
            filters: Optional metadata filters for payload filtering.
                Supports complex conditions using Qdrant's filter DSL.

        Returns:
            Dict with 'query' and 'documents' keys. Documents are ordered by
            cross-encoder relevance score in descending order.
        """
        # Stage 1: Embed query using bi-encoder for vector search
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Retrieve 3x candidates to give reranker sufficient options for selection
        retrieval_top_k = top_k * 3
        base_docs = self.db.search(
            query_vector=query_embedding,
            top_k=retrieval_top_k,
            filters=filters,
        )
        logger.info("Retrieved %d base documents", len(base_docs))

        if not base_docs:
            logger.warning("No base documents retrieved")
            return {"query": query, "documents": []}

        # Stage 2: Apply cross-encoder reranking for precision
        # Cross-encoder jointly encodes query and each document for fine-grained scoring
        reranked_result = self.reranker.run(query=query, documents=base_docs)
        reranked_docs = reranked_result.get("documents", [])[:top_k]

        logger.info("Reranked to %d documents", len(reranked_docs))
        return {"query": query, "documents": reranked_docs}

    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Alias for search() for backward compatibility with Haystack pipelines."""
        return self.search(query, top_k)
