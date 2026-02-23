"""Chroma reranking search pipeline (LangChain).

This module provides the search pipeline for Chroma vector database
with cross-encoder reranking. Chroma's local-first design makes this
ideal for development, testing, and offline search applications.

Reranking Process:
    1. Embed the query using the configured embedding model
    2. Retrieve top-k candidates from local Chroma collection
    3. Apply cross-encoder reranker to score query-document relevance
    4. Return top rerank_k documents ordered by cross-encoder scores
    5. Optionally generate RAG answer using LLM and reranked context

Key Features:
    - Local persistent storage without cloud dependencies
    - Fast candidate retrieval for development workflows
    - Optional RAG generation with configurable LLM
    - Simple setup ideal for prototyping reranking pipelines
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


logger = logging.getLogger(__name__)


class ChromaRerankingSearchPipeline:
    """Search pipeline with reranking for Chroma (LangChain).

    This pipeline implements two-stage retrieval with cross-encoder reranking
    on local Chroma collections, ideal for development and testing.

    Chroma's local-first architecture makes this pipeline perfect for:
    - Prototyping reranking applications
    - Offline development environments
    - Small to medium document collections
    - Testing cross-encoder configurations

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model for query encoding
        db: ChromaVectorDB instance for local storage
        collection_name: Name of the Chroma collection to query
        reranker: Cross-encoder reranker instance for scoring
        llm: Optional LLM for RAG answer generation

    Example:
        >>> pipeline = ChromaRerankingSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="machine learning tutorials", top_k=30, rerank_k=5
        ... )
        >>> print(f"Found {len(results['documents'])} reranked results")
        >>> for i, doc in enumerate(results["documents"], 1):
        ...     print(f"{i}. [{doc.score:.3f}] {doc.content[:60]}...")

    Note:
        Chroma stores data locally on disk. Ensure adequate storage
        space for your document collection and embeddings.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma reranking search pipeline.

        Loads configuration, initializes embedding model, reranker,
        and connects to local Chroma database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file with pipeline settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path", "./chroma_data"),
        )

        self.collection_name = chroma_config.get("collection_name", "reranking")
        self.reranker = RerankerHelper.create_reranker(self.config)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Chroma reranking search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search against Chroma collection.

        Performs two-stage retrieval: first retrieves candidates using
        vector similarity, then reranks using cross-encoder scores.

        Args:
            query: Search query text to execute.
            top_k: Number of candidates to retrieve before reranking.
                Higher values improve reranking quality but increase latency.
            rerank_k: Number of results to return after reranking.
                Should match your application's result display needs.
            filters: Optional metadata filters for pre-filtering candidates.
                Uses Chroma's where-document filtering syntax.

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
            2. Query Chroma collection for top_k candidate documents
            3. Apply cross-encoder reranker to score candidates
            4. Sort by reranker scores and return top rerank_k
            5. Generate RAG answer if LLM is configured
        """
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        self.db._get_collection(self.collection_name)
        results_dict = self.db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters,
        )
        candidates = self.db.query_to_documents(results_dict)
        logger.info("Retrieved %d candidate documents from Chroma", len(candidates))

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
