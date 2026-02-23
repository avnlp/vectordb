"""Weaviate reranking search pipeline.

This module provides a search pipeline combining Weaviate's rich query
capabilities with cross-encoder reranking. Weaviate's native support
for BM25, vector search, and filters makes it versatile for various
retrieval scenarios.

Reranking Strategy:
    1. Embed query using bi-encoder for vector representation
    2. Retrieve 3x candidates from Weaviate using dense search
    3. Score candidates with cross-encoder for precise relevance
    4. Return top_k results ordered by cross-encoder score

Weaviate Features:
    - Native hybrid search (combining BM25 and vector similarity)
    - GraphQL query interface with rich filtering
    - Module system for custom AI integrations
    - Multi-modal data support

Cross-Encoder vs Weaviate Hybrid:
    While Weaviate offers built-in hybrid search, cross-encoder reranking
    provides deeper semantic understanding by jointly encoding query and
    document, capturing subtle semantic relationships.
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory, RerankerFactory


logger = logging.getLogger(__name__)


class WeaviateRerankingSearchPipeline:
    """Weaviate search pipeline with cross-encoder reranking.

    Combines Weaviate's vector search capabilities with cross-encoder reranking
    for improved result precision. The pipeline retrieves candidates using
    dense embeddings, then applies expensive but accurate cross-encoder scoring.

    Attributes:
        config: Pipeline configuration with weaviate, embedder, reranker sections.
        embedder: Bi-encoder for query embedding and initial retrieval.
        reranker: Cross-encoder for final document scoring.
        db: WeaviateVectorDB instance connected to the Weaviate server.
        collection_name: Name of the Weaviate collection to search.

    Example:
        >>> pipeline = WeaviateRerankingSearchPipeline("weaviate_config.yaml")
        >>> results = pipeline.search(
        ...     query="attention mechanism in transformers", top_k=10
        ... )
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - weaviate: url, collection_name, vectorizer settings
                - embedder: Provider, model, dimensions for bi-encoder
                - reranker: Provider, model for cross-encoder

        Raises:
            ValueError: If required config sections are missing or invalid.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        self.reranker = RerankerFactory.create(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config.get("url", "http://localhost:8080"),
            collection_name=weaviate_config.get("collection_name", "Reranking"),
        )

        self.collection_name = weaviate_config.get("collection_name", "Reranking")

        logger.info("Initialized Weaviate reranking search pipeline")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search with two-stage retrieval.

        Stage 1 - Dense Retrieval:
            Embed query and retrieve candidates from Weaviate using vector
            similarity. Retrieves 3x the final top_k to ensure good coverage.

        Stage 2 - Cross-Encoder Reranking:
            Apply cross-encoder to score each candidate for relevance to query.
            Returns only the top_k highest-scoring documents.

        Args:
            query: Search query text to find relevant documents for.
            top_k: Number of final results to return after reranking.
            filters: Optional metadata filters for Weaviate's where clause.
                Supports path-based filtering on document properties.

        Returns:
            Dict with 'query' and 'documents' keys. Documents are sorted by
            cross-encoder relevance score in descending order.
        """
        # Stage 1: Embed query using bi-encoder
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Retrieve 3x candidates to give reranker sufficient options
        retrieval_top_k = top_k * 3
        base_docs = self.db.query(
            vector=query_embedding,
            limit=retrieval_top_k,
            filters=filters,
            return_documents=True,
        )
        logger.info("Retrieved %d base documents", len(base_docs))

        if not base_docs:
            logger.warning("No base documents retrieved")
            return {"query": query, "documents": []}

        # Stage 2: Apply cross-encoder reranking for precision
        # Cross-encoder scores query-document pairs jointly for accuracy
        reranked_result = self.reranker.run(query=query, documents=base_docs)
        reranked_docs = reranked_result.get("documents", [])[:top_k]

        logger.info("Reranked to %d documents", len(reranked_docs))
        return {"query": query, "documents": reranked_docs}

    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Alias for search() for backward compatibility with Haystack pipelines."""
        return self.search(query, top_k)
