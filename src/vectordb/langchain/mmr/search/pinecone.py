"""Pinecone MMR search pipeline (LangChain).

This module provides a LangChain-native MMR search pipeline for Pinecone
vector store indexes. MMR (Maximal Marginal Relevance) balances query
relevance with result diversity.

MMR Algorithm:
    The MMR algorithm scores each candidate document using the formula:
        MMR = λ * relevance(query, doc) - (1 - λ) * max_similarity(doc, selected_docs)

    Where:
        - λ (lambda) controls the trade-off between relevance and diversity
        - relevance is the similarity between query and document
        - max_similarity is the maximum similarity to already selected documents

Pipeline Architecture:
    1. Query Embedding: Convert query text to dense vector
    2. Candidate Retrieval: Fetch top_k most similar documents from Pinecone
    3. MMR Reranking: Apply diversity-aware selection to candidates
    4. Optional RAG: Generate answer using selected documents

Example:
    >>> from vectordb.langchain.mmr.search import PineconeMMRSearchPipeline
    >>> pipeline = PineconeMMRSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="What is machine learning?",
    ...     top_k=20,
    ...     mmr_k=5,
    ...     lambda_param=0.5,
    ... )

See Also:
    - vectordb.langchain.mmr.indexing.pinecone: Document indexing for Pinecone
    - vectordb.utils.mmr_helper: Core MMR algorithm implementation
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    MMRHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class PineconeMMRSearchPipeline:
    """LangChain MMR search pipeline for Pinecone.

    Implements diversity-aware document retrieval using the MMR algorithm
    on Pinecone vector store indexes. MMR reduces result redundancy while
    maintaining query relevance in production-scale deployments.

    Attributes:
        config: Validated configuration dictionary for Pinecone connection,
            embedder settings, and optional LLM configuration.
        embedder: LangChain embedder for query vectorization.
        db: PineconeVectorDB instance for vector search operations.
        index_name: Name of the Pinecone index to search.
        namespace: Pinecone namespace for document organization.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> config = {
        ...     "pinecone": {
        ...         "api_key": "pc-api-...",
        ...         "index_name": "mmr-docs",
        ...         "namespace": "production",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = PineconeMMRSearchPipeline(config)
        >>> results = pipeline.search("artificial intelligence", mmr_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone MMR search pipeline.

        Loads configuration, initializes the embedder, establishes connection
        to Pinecone, and optionally configures an LLM for RAG.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'pinecone' section with API key and index details.

        Raises:
            ValueError: If required Pinecone configuration (api_key) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Pinecone API.
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

        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Pinecone MMR search pipeline (LangChain)")

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
        similarity from Pinecone, then applies MMR to select mmr_k diverse
        documents from the candidate pool.

        The MMR algorithm greedily selects documents that maximize the
        MMR score, balancing query relevance against similarity to
        already-selected documents. This reduces redundancy in results.

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of candidate documents to retrieve before MMR.
                Higher values give MMR more options for diversity.
                Must be >= mmr_k.
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
            RuntimeError: If embedding generation or Pinecone query fails.

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

        candidates = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            namespace=self.namespace,
        )
        logger.info("Retrieved %d candidate documents from Pinecone", len(candidates))

        candidate_embeddings = []
        for doc in candidates:
            embedding = doc.metadata.get("embedding")
            if embedding is None:
                embedding = query_embedding
            candidate_embeddings.append(embedding)

        mmr_docs = MMRHelper.mmr_rerank_simple(
            candidates,
            candidate_embeddings,
            query_embedding,
            k=mmr_k,
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
