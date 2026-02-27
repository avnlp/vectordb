"""Weaviate diversity filtering search pipeline (LangChain).

This module provides a search pipeline for Weaviate vector database with
diversity filtering capabilities. Diversity filtering post-processes search
results to ensure the returned documents cover different aspects of the query,
reducing redundancy and improving information coverage.

Diversity Filtering Methods:
    1. Threshold-based (default):
       - Iteratively selects documents that are most relevant to the query
       - Filters out documents with similarity > threshold to already-selected docs
       - Configurable: max_documents, similarity_threshold (default: 0.7)
       - Best for: Fine-grained control over diversity vs relevance trade-off

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
    2. Over-fetch: Retrieve 3x top_k candidates from Weaviate
    3. Re-embedding: Generate embeddings for retrieved documents
    4. Diversity Filtering: Apply threshold or clustering method
    5. Limit: Return top_k diverse documents
    6. Optional RAG: Generate answer using diverse documents

Configuration Schema:
    Required:
        weaviate.url: Weaviate server URL
        weaviate.collection_name: Target collection name
    Optional:
        weaviate.api_key: API key for authentication
        diversity.method: "threshold" or "clustering"
        diversity.max_documents: Max docs for threshold method
        diversity.similarity_threshold: Similarity cutoff (0.0-1.0)
        diversity.num_clusters: Number of clusters for clustering method
        diversity.samples_per_cluster: Docs per cluster
        rag: Optional LLM configuration for answer generation

Example:
    >>> from vectordb.langchain.diversity_filtering.search import (
    ...     WeaviateDiversityFilteringSearchPipeline,
    ... )
    >>> pipeline = WeaviateDiversityFilteringSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="machine learning applications",
    ...     top_k=5,
    ... )
    >>> for doc in results["documents"]:
    ...     print(f"Diverse result: {doc.page_content[:100]}...")

See Also:
    - vectordb.langchain.diversity_filtering.indexing.weaviate: Document indexing
    - vectordb.utils.diversification_helper: Diversity algorithm implementations
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    DiversificationHelper,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class WeaviateDiversityFilteringSearchPipeline:
    """Weaviate search pipeline with diversity filtering (LangChain).

    Implements diversity-aware document retrieval by over-fetching candidates
    from Weaviate and applying post-processing to select diverse results.
    Supports both threshold-based and clustering-based diversity methods.

    Attributes:
        config: Validated configuration dictionary for Weaviate connection,
            embedder settings, diversity parameters, and optional LLM.
        embedder: LangChain embedder for query and document vectorization.
        db: WeaviateVectorDB instance for vector search operations.
        collection_name: Name of the Weaviate collection to search.
        diversity_config: Configuration for diversity filtering parameters.
        method: Diversity method - "threshold" or "clustering".
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
        ...     "weaviate": {"url": "http://localhost:8080", "collection_name": "docs"},
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "diversity": {"method": "threshold", "similarity_threshold": 0.7},
        ... }
        >>> pipeline = WeaviateDiversityFilteringSearchPipeline(config)
        >>> results = pipeline.search("AI ethics", top_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate diversity filtering search pipeline.

        Loads configuration, initializes the embedder, establishes connection
        to Weaviate, and configures diversity filtering parameters.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'weaviate' section with URL and collection details.

        Raises:
            ValueError: If required Weaviate configuration (url) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Weaviate API.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        # Initialize embedder for query and document vectorization.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Establish connection to Weaviate vector database.
        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        # Store Weaviate settings for search operations.
        self.collection_name = weaviate_config.get("collection_name")

        # Configure diversity filtering parameters.
        # Defaults to threshold-based method with sensible defaults.
        self.diversity_config = self.config.get("diversity", {})
        self.method = self.diversity_config.get("method", "threshold")

        # Initialize optional LLM for RAG answer generation.
        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Weaviate diversity filtering search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute diversity filtering search.

        Performs semantic search with post-processing diversity filtering.
        Over-fetches candidates from Weaviate, then applies either threshold-based
        or clustering-based diversity selection to return diverse results.

        The diversity filtering process:
            1. Retrieve 3x top_k candidates from Weaviate
            2. Re-embed retrieved documents for consistency
            3. Apply selected diversity method (threshold or clustering)
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
            RuntimeError: If embedding generation or Weaviate query fails.

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
        retrieved_documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k * 3,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d documents from Weaviate", len(retrieved_documents))

        # Re-embed retrieved documents to ensure consistent embeddings.
        # Stored embeddings might use a different model or version.
        doc_texts = [doc.page_content for doc in retrieved_documents]
        doc_embeddings = [
            EmbedderHelper.embed_query(self.embedder, text) for text in doc_texts
        ]

        # Apply diversity filtering based on configured method.
        if self.method == "threshold":
            # Threshold method: greedily select documents, filtering out those
            # too similar to already-selected ones.
            max_documents = self.diversity_config.get("max_documents", top_k)
            similarity_threshold = self.diversity_config.get(
                "similarity_threshold", 0.7
            )
            diverse_documents = DiversificationHelper.diversify(
                documents=retrieved_documents,
                embeddings=doc_embeddings,
                max_documents=max_documents,
                similarity_threshold=similarity_threshold,
            )
        else:  # clustering method
            # Clustering method: group documents into clusters, then sample
            # from each cluster to ensure topic coverage.
            num_clusters = self.diversity_config.get("num_clusters", 3)
            samples_per_cluster = self.diversity_config.get(
                "samples_per_cluster", top_k // 3
            )
            diverse_documents = DiversificationHelper.clustering_based_diversity(
                documents=retrieved_documents,
                embeddings=doc_embeddings,
                num_clusters=num_clusters,
                samples_per_cluster=samples_per_cluster,
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
