"""Milvus reranking search pipeline (LangChain).

This module provides the search pipeline for Milvus vector database
with cross-encoder reranking. Milvus's GPU acceleration and billion-
scale capacity make it ideal for enterprise reranking deployments.

Reranking Process:
    1. Embed the query using the configured embedding model
    2. Retrieve top-k candidates using Milvus's advanced indexing
    3. Apply cross-encoder reranker to score query-document relevance
    4. Return top rerank_k documents ordered by cross-encoder scores
    5. Optionally generate RAG answer using LLM and reranked context

Key Features:
    - GPU-accelerated embedding and search
    - Billion-scale vector collection support
    - Advanced indexing (IVF, HNSW, ANNOY, etc.)
    - Partitioning for data organization and performance
    - Optional RAG generation with configurable LLM
    - Enterprise-grade high availability
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


logger = logging.getLogger(__name__)


class MilvusReankingSearchPipeline:
    """Search pipeline with reranking for Milvus (LangChain).

    This pipeline implements two-stage retrieval with cross-encoder reranking
    on Milvus collections, designed for large-scale enterprise applications.

    Milvus is ideal for reranking requiring:
    - Massive scale (billions of vectors)
    - GPU acceleration for compute-intensive operations
    - Advanced indexing algorithm selection
    - Data partitioning strategies
    - Enterprise high availability

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model for query encoding
        db: MilvusVectorDB instance for database operations
        collection_name: Name of the Milvus collection
        reranker: Cross-encoder reranker instance for scoring
        llm: Optional LLM for RAG answer generation

    Example:
        >>> pipeline = MilvusReankingSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="deep learning optimization techniques", top_k=200, rerank_k=20
        ... )
        >>> print(f"Retrieved and reranked {len(results['documents'])} documents")
        >>> for doc in results["documents"][:3]:
        ...     print(f"Score {doc.score:.3f}: {doc.page_content[:80]}...")

    Enterprise Features:
        - Supports multiple index types (IVF_FLAT, HNSW, etc.)
        - Partitioning for parallel query execution
        - Role-based access control (RBAC)
        - Multi-replica for high availability
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus reranking search pipeline.

        Loads configuration, initializes embedding model, reranker,
        and connects to Milvus server.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file with pipeline settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Milvus.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get("collection_name", "reranking")
        self.reranker = RerankerHelper.create_reranker(self.config)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Milvus reranking search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search against Milvus collection.

        Performs two-stage retrieval: first retrieves candidates using
        vector similarity, then reranks using cross-encoder scores.

        Args:
            query: Search query text to execute.
            top_k: Number of candidates to retrieve before reranking.
                Higher values improve reranking quality but increase latency.
            rerank_k: Number of results to return after reranking.
                Should match your application's result display needs.
            filters: Optional metadata filters for pre-filtering candidates.
                Uses Milvus's expression syntax.

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
            2. Query Milvus for top_k candidate documents
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
        logger.info("Retrieved %d candidate documents from Milvus", len(candidates))

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
