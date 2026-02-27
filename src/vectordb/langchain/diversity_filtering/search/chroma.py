"""Chroma diversity filtering search pipeline (LangChain).

This module provides a search pipeline for Chroma vector database with
diversity filtering capabilities. Diversity filtering post-processes search
results to ensure the returned documents cover different aspects of the query,
reducing redundancy and improving information coverage.

Diversity Filtering Methods:
    1. MMR - Maximal Marginal Relevance (default):
       - Balances query relevance with inter-document diversity
       - Uses lambda parameter to control relevance-diversity trade-off
       - Configurable: max_documents, lambda_param (default: 0.5)
       - Best for: Retrieval where both relevance and diversity matter

    2. Clustering-based:
       - Groups retrieved documents into N clusters using embeddings
       - Samples M documents from each cluster
       - Configurable: num_clusters, samples_per_cluster
       - Best for: Ensuring coverage of distinct topic areas

Why Diversity Matters:
    Standard semantic search returns the k most similar documents, which often
    results in redundant information (e.g., 5 similar paragraphs from the same
    source). Diversity filtering ensures results cover different perspectives,
    sources, or aspects of the query topic.

Pipeline Architecture:
    1. Query Embedding: Convert query text to dense vector
    2. Over-fetch: Retrieve 3x top_k candidates from Chroma
    3. Re-embedding: Generate embeddings for retrieved documents
    4. Diversity Filtering: Apply MMR or clustering method
    5. Limit: Return top_k diverse documents
    6. Optional RAG: Generate answer using diverse documents

Configuration Schema:
    Required:
        chroma.persist_dir: Directory for Chroma persistence
        chroma.collection_name: Target collection name
    Optional:
        diversity.method: "mmr" or "clustering"
        diversity.max_documents: Max docs for MMR method
        diversity.lambda_param: Relevance-diversity trade-off (0.0-1.0)
        diversity.num_clusters: Number of clusters for clustering method (default: 3)
        diversity.samples_per_cluster: Docs per cluster (default: 2)
        diversity.candidate_multiplier: Over-fetch multiplier (default: 3)
        rag: Optional LLM configuration for answer generation

Example:
    >>> from vectordb.langchain.diversity_filtering.search import (
    ...     ChromaDiversityFilteringSearchPipeline,
    ... )
    >>> pipeline = ChromaDiversityFilteringSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="machine learning applications",
    ...     top_k=5,
    ... )
    >>> for doc in results["documents"]:
    ...     print(f"Diverse result: {doc.page_content[:100]}...")

See Also:
    - vectordb.langchain.diversity_filtering.indexing.chroma: Document indexing
    - vectordb.langchain.diversity_filtering.helpers:
      Diversity algorithm implementations
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.diversity_filtering.helpers import DiversityFilteringHelper
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class ChromaDiversityFilteringSearchPipeline:
    """Chroma search pipeline with diversity filtering (LangChain).

    Implements diversity-aware document retrieval by over-fetching candidates
    from Chroma and applying post-processing to select diverse results.
    Supports both MMR and clustering-based diversity methods.

    Attributes:
        config: Validated configuration dictionary for Chroma connection,
            embedder settings, diversity parameters, and optional LLM.
        embedder: LangChain embedder for query and document vectorization.
        db: ChromaVectorDB instance for vector search operations.
        collection_name: Name of the Chroma collection to search.
        diversity_config: Configuration for diversity filtering parameters.
        method: Diversity method - "mmr" or "clustering".
        llm: Optional LangChain LLM for RAG answer generation.

    Design Decisions:
        - Over-fetching: Retrieves 3x top_k documents to give diversity algorithm
          more options for selecting diverse results.
        - Re-embedding: Re-embeds retrieved documents to ensure consistent
          embeddings for diversity calculations (in case stored embeddings differ).
        - Post-processing: Diversity is applied after retrieval, allowing
          flexible parameters without re-indexing.

    Example:
        >>> config = {
        ...     "chroma": {"persist_dir": "./chroma_data", "collection_name": "docs"},
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "diversity": {"method": "mmr", "lambda_param": 0.5},
        ... }
        >>> pipeline = ChromaDiversityFilteringSearchPipeline(config)
        >>> results = pipeline.search("AI ethics", top_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma diversity filtering search pipeline.

        Loads configuration, initializes the embedder, establishes connection
        to Chroma, and configures diversity filtering parameters.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'chroma' section with persist_dir and collection details.

        Raises:
            ValueError: If required Chroma configuration (persist_dir) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Chroma database.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        # Initialize embedder for query and document vectorization.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Establish connection to Chroma vector database.
        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_dir=chroma_config.get("persist_dir"),
        )

        # Store Chroma settings for search operations.
        self.collection_name = chroma_config.get("collection_name")

        # Configure diversity filtering parameters.
        # Defaults to MMR with sensible defaults.
        self.diversity_config = self.config.get("diversity", {})
        self.method = self.diversity_config.get("method", "mmr")

        # Initialize optional LLM for RAG answer generation.
        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Chroma diversity filtering search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute diversity filtering search.

        Performs semantic search with post-processing diversity filtering.
        Over-fetches candidates from Chroma, then applies either MMR or
        clustering-based diversity selection to return diverse results.

        The diversity filtering process:
            1. Retrieve 3x top_k candidates from Chroma
            2. Re-embed retrieved documents for consistency
            3. Apply selected diversity method (MMR or clustering)
            4. Limit results to top_k diverse documents
            5. Optionally generate RAG answer

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of diverse documents to return. Default is 10.
                The pipeline retrieves 3x this amount for diversity selection.
            filters: Optional metadata filters to apply during retrieval.
                Dictionary mapping field names to filter values.

        Returns:
            Dictionary containing:
                - documents: List of diverse Document objects (max top_k)
                - query: Original query string
                - answer: Generated RAG answer if LLM configured (optional)

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation or Chroma query fails.

        Example:
            >>> results = pipeline.search(
            ...     query="renewable energy sources",
            ...     top_k=5,
            ...     filters={"category": "technology"},
            ... )
            >>> print(f"Found {len(results['documents'])} diverse documents")
        """
        logger.info("Starting diversity filtering search (method=%s)", self.method)

        # Generate query embedding for semantic search.
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Over-fetch candidates to give diversity algorithm more options.
        # 3x multiplier provides enough candidates for effective diversity selection.
        candidate_multiplier = self.diversity_config.get("candidate_multiplier", 3)
        retrieved_documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k * candidate_multiplier,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d documents from Chroma", len(retrieved_documents))

        # Re-embed retrieved documents to ensure consistent embeddings.
        # Stored embeddings might use a different model or version.
        doc_texts = [doc.page_content for doc in retrieved_documents]
        doc_embeddings = [
            EmbedderHelper.embed_query(self.embedder, text) for text in doc_texts
        ]

        # Apply diversity filtering based on configured method.
        if self.method == "clustering":
            # Clustering method: group documents into clusters, then sample
            # from each cluster to ensure topic coverage.
            num_clusters = self.diversity_config.get("num_clusters", 3)
            samples_per_cluster = self.diversity_config.get("samples_per_cluster", 2)
            diverse_documents = DiversityFilteringHelper.clustering_diversify(
                documents=retrieved_documents,
                embeddings=doc_embeddings,
                num_clusters=num_clusters,
                samples_per_cluster=samples_per_cluster,
            )
        elif self.method == "mmr":
            # MMR method: balance relevance to the query with novelty against
            # already-selected documents.
            max_documents = self.diversity_config.get("max_documents", top_k)
            lambda_param = self.diversity_config.get("lambda_param", 0.5)
            diverse_documents = DiversityFilteringHelper.mmr_diversify(
                documents=retrieved_documents,
                embeddings=doc_embeddings,
                query_embedding=query_embedding,
                max_documents=max_documents,
                lambda_param=lambda_param,
            )
        else:
            raise ValueError(
                f"Unknown diversity method: {self.method}. "
                "Expected one of: ['mmr', 'clustering']"
            )

        # Limit to requested top_k (diversity methods may return more).
        diverse_documents = diverse_documents[:top_k]
        logger.info("Diversity filtered to %d documents", len(diverse_documents))

        result = {
            "documents": diverse_documents,
            "query": query,
        }

        # Generate RAG answer if LLM is configured.
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, diverse_documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
