"""Weaviate reranking search pipeline (LangChain).

This module provides the search pipeline for Weaviate vector database
with cross-encoder reranking. Weaviate's AI-native design with GraphQL
interface integrates seamlessly with reranking workflows.

Reranking Process:
    1. Embed the query using the configured embedding model
    2. Retrieve top-k candidates using Weaviate's vector search
    3. Apply cross-encoder reranker to score query-document relevance
    4. Return top rerank_k documents ordered by cross-encoder scores
    5. Optionally generate RAG answer using LLM and reranked context

Key Features:
    - Native GraphQL interface for flexible querying
    - Hybrid search combining vector and BM25 rankings
    - Built-in vectorization modules (optional)
    - Optional RAG generation with configurable LLM
    - Rich property/metadata support for filtering
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


logger = logging.getLogger(__name__)


class WeaviateReankingSearchPipeline:
    """Search pipeline with reranking for Weaviate (LangChain).

    This pipeline implements two-stage retrieval with cross-encoder reranking
    on Weaviate collections, leveraging Weaviate's AI-native architecture.

    Weaviate is ideal for reranking applications requiring:
    - GraphQL query interface
    - Hybrid vector + keyword search
    - Cloud or self-hosted deployment
    - Rich data modeling with properties

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model for query encoding
        db: WeaviateVectorDB instance for database operations
        collection_name: Name of the Weaviate class/collection
        reranker: Cross-encoder reranker instance for scoring
        llm: Optional LLM for RAG answer generation

    Example:
        >>> pipeline = WeaviateReankingSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="climate change effects",
        ...     top_k=40,
        ...     rerank_k=8,
        ...     filters={"category": "environment"},
        ... )
        >>> for doc in results["documents"]:
        ...     print(f"Rerank Score: {doc.score:.4f}")
        ...     print(f"Content: {doc.page_content[:100]}...\n")

    Weaviate-Specific Notes:
        - Collection names are capitalized by convention in Weaviate
        - Supports hybrid search with alpha parameter (not used in reranking)
        - Can leverage Weaviate's native reranking modules alternatively
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate reranking search pipeline.

        Loads configuration, initializes embedding model, reranker,
        and connects to Weaviate instance.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file with pipeline settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Weaviate.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name", "Reranking")
        self.reranker = RerankerHelper.create_reranker(self.config)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Weaviate reranking search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search against Weaviate collection.

        Performs two-stage retrieval: first retrieves candidates using
        vector similarity, then reranks using cross-encoder scores.

        Args:
            query: Search query text to execute.
            top_k: Number of candidates to retrieve before reranking.
                Higher values improve reranking quality but increase latency.
            rerank_k: Number of results to return after reranking.
                Should match your application's result display needs.
            filters: Optional metadata filters for pre-filtering candidates.
                Uses Weaviate's where filter syntax.

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
            2. Query Weaviate for top_k candidate documents
            3. Apply cross-encoder reranker to score candidates
            4. Sort by reranker scores and return top rerank_k
            5. Generate RAG answer if LLM is configured
        """
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        candidates = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d candidate documents from Weaviate", len(candidates))

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
