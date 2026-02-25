"""Qdrant reranking search pipeline (LangChain).

This module provides the search pipeline for Qdrant vector database
with cross-encoder reranking. Qdrant's high-performance architecture
makes it ideal for production reranking applications.

Reranking Process:
    1. Embed the query using the configured embedding model
    2. Retrieve top-k candidates using Qdrant's HNSW index
    3. Apply cross-encoder reranker to score query-document relevance
    4. Return top rerank_k documents ordered by cross-encoder scores
    5. Optionally generate RAG answer using LLM and reranked context

Key Features:
    - High-performance HNSW indexing for fast candidate retrieval
    - Rich payload filtering with complex conditions
    - Efficient memory usage and query performance
    - Optional RAG generation with configurable LLM
    - Support for both local and remote Qdrant deployments
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    HaystackToLangchainConverter,
    RAGHelper,
    RerankerHelper,
)


logger = logging.getLogger(__name__)


class QdrantRerankingSearchPipeline:
    """Search pipeline with reranking for Qdrant (LangChain).

    This pipeline implements two-stage retrieval with cross-encoder reranking
    on Qdrant collections, optimized for high-performance applications.

    Qdrant is ideal for reranking requiring:
    - Fast approximate nearest neighbor search with HNSW
    - Complex filtering on payload fields
    - Cost-effective self-hosted deployments
    - Hybrid local/remote deployment flexibility

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model for query encoding
        db: QdrantVectorDB instance for database operations
        collection_name: Name of the Qdrant collection
        reranker: Cross-encoder reranker instance for scoring
        llm: Optional LLM for RAG answer generation

    Example:
        >>> pipeline = QdrantRerankingSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="neural network architectures",
        ...     top_k=100,
        ...     rerank_k=15,
        ...     filters={"must": [{"key": "topic", "match": {"value": "AI"}}]},
        ... )
        >>> print(f"Query: {results['query']}")
        >>> print(f"Top reranked result: {results['documents'][0].page_content[:200]}")

    Performance Notes:
        - Qdrant's HNSW index provides fast initial retrieval
        - Payload filters are applied during vector search
        - Cross-encoder scoring occurs after retrieval
        - Consider Qdrant's built-in reranking for simpler use cases
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Qdrant reranking search pipeline.

        Loads configuration, initializes embedding model, reranker,
        and connects to Qdrant server.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file with pipeline settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Qdrant.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name", "reranking")
        self.reranker = RerankerHelper.create_reranker(self.config)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Qdrant reranking search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search against Qdrant collection.

        Performs two-stage retrieval: first retrieves candidates using
        vector similarity, then reranks using cross-encoder scores.

        Args:
            query: Search query text to execute.
            top_k: Number of candidates to retrieve before reranking.
                Higher values improve reranking quality but increase latency.
            rerank_k: Number of results to return after reranking.
                Should match your application's result display needs.
            filters: Optional payload filters for pre-filtering candidates.
                Uses Qdrant's filter syntax with must/should/must_not.

        Returns:
            Dictionary containing search results:
            - documents: List of reranked documents with cross-encoder scores
            - query: The original search query
            - answer: Optional RAG-generated answer (if LLM configured)

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If search or reranking fails.

        Search Process:
            1. Embed query using configured embedding model
            2. Query Qdrant for top_k candidate documents
            3. Apply cross-encoder reranker to score candidates
            4. Sort by reranker scores and return top rerank_k
            5. Generate RAG answer if LLM is configured
        """
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        candidates = self.db.search(
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        candidates = HaystackToLangchainConverter.convert(candidates)
        logger.info("Retrieved %d candidate documents from Qdrant", len(candidates))

        reranked_docs = RerankerHelper.rerank(
            self.reranker,
            query,
            candidates,
            top_k=rerank_k,
        )
        logger.info("Reranked to %d documents", len(reranked_docs))

        result = {
            "documents": reranked_docs,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, reranked_docs)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
