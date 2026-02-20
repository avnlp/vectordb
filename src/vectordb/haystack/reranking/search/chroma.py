"""Chroma reranking search pipeline.

This module provides a search pipeline that combines Chroma's local or server-based
vector similarity search with cross-encoder reranking to improve result quality.

Reranking Fundamentals:
    Reranking addresses a key limitation of vector search: bi-encoders encode
    queries and documents independently, missing query-document interactions.
    Cross-encoders process pairs jointly, capturing nuanced relevance signals.

Two-Stage Retrieval Process:
    1. Bi-encoder embedding → Fast approximate nearest neighbor search in Chroma
    2. Cross-encoder scoring → Precise relevance assessment on top candidates

Retrieval Multiplier:
    The pipeline retrieves 3x the requested top_k to ensure the reranker has
    sufficient candidates. This balances:
    - Coverage: Higher chance of including truly relevant documents
    - Cost: Reranking 3x documents is still much faster than exhaustive search
    - Latency: Acceptable overhead for precision gains

When to Use This Pipeline:
    - High-precision requirements (question answering, semantic search)
    - Local development or self-hosted deployments (Chroma)
    - When latency can tolerate 100-500ms reranking overhead
    - Documents require deep semantic understanding beyond keyword matching

Example:
    >>> pipeline = ChromaRerankingSearchPipeline("config.yaml")
    >>> results = pipeline.search("transformer attention mechanism", top_k=5)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.score}, Content: {doc.content[:100]}...")
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory, RerankerFactory


logger = logging.getLogger(__name__)


class ChromaRerankingSearchPipeline:
    """Chroma search pipeline with cross-encoder reranking.

    Implements two-stage retrieval using Chroma for fast vector similarity
    search and a cross-encoder for precise final ranking. This approach
    delivers high-precision results while maintaining acceptable latency.

    Architecture:
        Stage 1 (Fast): Bi-encoder embeds query; Chroma retrieves 3x candidates
        Stage 2 (Precise): Cross-encoder scores each query-document pair;
        Top K results returned

    Attributes:
        config: Configuration dict with chroma, embedder, and reranker settings.
        embedder: Bi-encoder component for query embedding generation.
        reranker: Cross-encoder component for document scoring and reordering.
        db: ChromaVectorDB instance for document storage and retrieval.
        collection_name: Name of the Chroma collection being searched.

    Performance Characteristics:
        - Vector search: O(log n) with HNSW index (n = document count)
        - Reranking: O(k) where k = 3 * top_k (fixed small number)
        - Total latency: ~50-200ms for vector search + 100-500ms for reranking
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - chroma: host, port, collection_name settings
                - embedder: Provider (openai, sentence-transformers, etc.),
                  model name, dimensions, batch_size
                - reranker: Provider, model name for cross-encoder

        Raises:
            ValueError: If required configuration sections are missing.
            ConnectionError: If Chroma server is unreachable (when using server mode).
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        self.reranker = RerankerFactory.create(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            host=chroma_config.get("host", "localhost"),
            port=chroma_config.get("port", 8000),
            collection_name=chroma_config.get("collection_name", "reranking"),
        )

        self.collection_name = chroma_config.get("collection_name", "reranking")

        logger.info("Initialized Chroma reranking search pipeline")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search with two-stage retrieval.

        Performs dense retrieval followed by cross-encoder reranking to deliver
        high-precision search results.

        Stage 1 - Dense Retrieval:
            The query is embedded using a bi-encoder, then Chroma's HNSW index
            is queried for approximate nearest neighbors. We retrieve 3x the
            requested top_k to ensure good coverage for reranking.

        Stage 2 - Cross-Encoder Reranking:
            Each candidate document is scored against the query using a cross-
            encoder model. This joint encoding captures semantic nuances missed
            by independent bi-encoder embeddings. Results are sorted by score.

        Args:
            query: Search query text to find relevant documents for.
            top_k: Number of final results to return after reranking.
                The pipeline retrieves 3x this amount initially to maximize
                the chance of finding truly relevant documents.
            filters: Optional metadata filters to apply during vector search.
                Chroma supports filtering on document metadata fields using
                equality, range, and logical operators.

        Returns:
            Dict with 'query' (original query string) and 'documents' (list of
            Document objects sorted by cross-encoder relevance score, highest
            first). Each document includes score, content, and metadata.

        Raises:
            RuntimeError: If embedder or reranker fails during execution.
        """
        # Stage 1: Embed the query using the text embedder
        # The bi-encoder converts text to dense vector representation
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Stage 2: Retrieve candidates using vector similarity
        # Retrieve 3x more than requested to give reranker sufficient candidates
        # This balances: coverage (find relevant docs) vs cost (reranking time)
        retrieval_top_k = top_k * 3
        base_docs = self.db.search(
            query_dense_embedding=query_embedding,
            top_k=retrieval_top_k,
            filters=filters,
        )
        logger.info("Retrieved %d base documents", len(base_docs))

        # Handle empty results gracefully
        if not base_docs:
            logger.warning("No base documents retrieved")
            return {"query": query, "documents": []}

        # Stage 3: Rerank candidates using cross-encoder
        # Cross-encoder scores each query-document pair jointly, providing
        # more accurate relevance assessment than bi-encoder cosine similarity
        reranked_result = self.reranker.run(query=query, documents=base_docs)
        reranked_docs = reranked_result.get("documents", [])[:top_k]

        logger.info("Reranked to %d documents", len(reranked_docs))
        return {"query": query, "documents": reranked_docs}

    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Alias for search() for backward compatibility with Haystack pipelines."""
        return self.search(query, top_k)
