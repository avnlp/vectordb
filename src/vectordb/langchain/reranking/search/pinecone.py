"""Pinecone reranking search pipeline (LangChain).

This module provides the search pipeline for Pinecone vector database
with cross-encoder reranking. It implements a two-stage retrieval process
to improve search relevance beyond pure vector similarity.

Reranking Process:
    1. Embed the query using the configured embedding model
    2. Retrieve top-k candidates using fast ANN search in Pinecone
    3. Apply cross-encoder reranker to score query-document relevance
    4. Return top rerank_k documents ordered by cross-encoder scores
    5. Optionally generate RAG answer using LLM and reranked context

Key Features:
    - Metadata filtering support for pre-filtering candidates
    - Namespace isolation for multi-tenant scenarios
    - Optional RAG generation with configurable LLM
    - Efficient cloud-native vector search with reranking refinement
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


logger = logging.getLogger(__name__)


class PineconeReankingSearchPipeline:
    """Search pipeline with reranking for Pinecone (LangChain).

    This pipeline implements two-stage retrieval with cross-encoder reranking
    to improve search quality over pure vector similarity search.

    The reranking approach leverages cross-encoder models that process
    query-document pairs together, enabling deeper semantic understanding
    than bi-encoder embeddings alone.

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model for query encoding
        db: PineconeVectorDB instance for candidate retrieval
        index_name: Name of the Pinecone index to query
        namespace: Namespace for document isolation
        reranker: Cross-encoder reranker instance for scoring
        llm: Optional LLM for RAG answer generation

    Example:
        >>> pipeline = PineconeReankingSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="renewable energy technologies",
        ...     top_k=50,
        ...     rerank_k=10,
        ...     filters={"category": "science"},
        ... )
        >>> print(f"Query: {results['query']}")
        >>> for doc in results["documents"]:
        ...     print(f"Score: {doc.score:.3f} - {doc.content[:80]}...")
        >>> if "answer" in results:
        ...     print(f"RAG Answer: {results['answer']}")

    Performance Considerations:
        - top_k controls the candidate pool size; larger values improve
          reranking quality but increase latency
        - rerank_k should match your UI's result display count
        - Cross-encoder latency scales linearly with top_k
        - Consider using smaller cross-encoder models (e.g., 6-layer)
          for latency-sensitive applications
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone reranking search pipeline.

        Loads configuration, initializes embedding model, reranker,
        and establishes connection to Pinecone.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file with pipeline settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Pinecone.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        self.reranker = RerankerHelper.create_reranker(self.config)

        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Pinecone reranking search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search against Pinecone index.

        Performs two-stage retrieval: first retrieves candidates using
        vector similarity, then reranks using cross-encoder scores.

        Args:
            query: Search query text to execute.
            top_k: Number of candidates to retrieve before reranking.
                Higher values improve reranking quality but increase latency.
            rerank_k: Number of results to return after reranking.
                Should match your application's result display needs.
            filters: Optional metadata filters for pre-filtering candidates.
                Uses Pinecone's metadata filtering syntax.

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
            2. Query Pinecone for top_k candidate documents
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
            namespace=self.namespace,
        )
        logger.info("Retrieved %d candidate documents from Pinecone", len(candidates))

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
