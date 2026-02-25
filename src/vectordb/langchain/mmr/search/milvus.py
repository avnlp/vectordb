"""Milvus MMR search pipeline (LangChain).

This module provides a LangChain-native MMR search pipeline for Milvus
vector store collections. MMR (Maximal Marginal Relevance) balances
query relevance with result diversity.

MMR Algorithm:
    The MMR algorithm scores each candidate document using the formula:
        MMR = λ * relevance(query, doc) - (1 - λ) * max_similarity(doc, selected_docs)

    Where:
        - λ (lambda) controls the trade-off between relevance and diversity
        - relevance is the similarity between query and document
        - max_similarity is the maximum similarity to already selected documents

Pipeline Architecture:
    1. Query Embedding: Convert query text to dense vector
    2. Candidate Retrieval: Fetch top_k most similar documents from Milvus
    3. MMR Reranking: Apply diversity-aware selection to candidates
    4. Optional RAG: Generate answer using selected documents

Example:
    >>> from vectordb.langchain.mmr.search import MilvusMMRSearchPipeline
    >>> pipeline = MilvusMMRSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="What is machine learning?",
    ...     top_k=20,
    ...     mmr_k=5,
    ...     lambda_param=0.5,
    ... )

See Also:
    - vectordb.langchain.mmr.indexing.milvus: Document indexing for Milvus
    - vectordb.utils.mmr_helper: Core MMR algorithm implementation
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    HaystackToLangchainConverter,
    MMRHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class MilvusMMRSearchPipeline:
    """LangChain MMR search pipeline for Milvus.

    Implements diversity-aware document retrieval using the MMR algorithm
    on Milvus vector store collections. MMR reduces result redundancy while
    maintaining query relevance.

    Attributes:
        config: Validated configuration dictionary for Milvus connection,
            embedder settings, and optional LLM configuration.
        embedder: LangChain embedder for query vectorization.
        db: MilvusVectorDB instance for vector search operations.
        collection_name: Name of the Milvus collection to search.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> config = {
        ...     "milvus": {
        ...         "host": "localhost",
        ...         "port": 19530,
        ...         "collection_name": "mmr",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = MilvusMMRSearchPipeline(config)
        >>> results = pipeline.search("artificial intelligence", mmr_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus MMR search pipeline.

        Loads configuration, initializes the embedder, establishes connection
        to Milvus, and optionally configures an LLM for RAG.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'milvus' section with connection details.

        Raises:
            ValueError: If required Milvus configuration is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Milvus server.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get("collection_name", "mmr")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Milvus MMR search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        mmr_k: int = 5,
        lambda_param: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute MMR search with diversity-aware document selection.

        Performs a two-stage search: first retrieves top_k candidates by
        similarity, then applies MMR to select mmr_k diverse documents.

        The MMR algorithm greedily selects documents that maximize the
        MMR score, balancing query relevance against similarity to
        already-selected documents.

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of candidate documents to retrieve before MMR.
                Higher values give MMR more options for diversity.
            mmr_k: Number of documents to return after MMR selection.
                Must be less than or equal to top_k.
            lambda_param: Trade-off parameter between relevance and diversity.
                Value of 1.0 prioritizes relevance only, 0.0 prioritizes
                diversity only. Default is 0.5 for balanced selection.
            filters: Optional metadata filters to apply during retrieval.
                Dictionary mapping field names to filter values.

        Returns:
            Dictionary containing:
                - documents: List of mmr_k selected Document objects
                - query: Original query string
                - answer: Generated RAG answer if LLM configured (optional)

        Raises:
            ValueError: If mmr_k > top_k or if query is empty.
            RuntimeError: If embedding generation or Milvus query fails.

        Example:
            >>> results = pipeline.search(
            ...     query="neural networks",
            ...     top_k=20,
            ...     mmr_k=5,
            ...     lambda_param=0.6,
            ... )
            >>> print(f"Found {len(results['documents'])} diverse documents")
        """
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        haystack_candidates = self.db.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
            include_vectors=True,
        )
        logger.info(
            "Retrieved %d candidate documents from Milvus", len(haystack_candidates)
        )

        candidates, candidate_embeddings = (
            HaystackToLangchainConverter.convert_with_embeddings(haystack_candidates)
        )

        mmr_docs = MMRHelper.mmr_rerank_simple(
            candidates,
            candidate_embeddings,
            query_embedding,
            k=mmr_k,
            lambda_param=lambda_param,
        )
        logger.info("Applied MMR to %d documents", len(mmr_docs))

        result = {
            "documents": mmr_docs,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, mmr_docs)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
