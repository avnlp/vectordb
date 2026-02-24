"""Weaviate cost-optimized RAG search pipeline for LangChain.

This module implements a cost-optimized search pipeline for Weaviate that
leverages native hybrid search combining dense semantic vectors with BM25
lexical search.

Cost Optimization:
    - Single dense query embedding: One API call for semantic search
    - Local sparse embedding: TF-IDF generated locally (zero cost)
    - Weaviate native hybrid search: Built-in BM25 + vector fusion
    - Optional LLM: RAG generation can be disabled

Weaviate Features:
    Weaviate provides native hybrid search with BM25 lexical matching,
    combining dense vector similarity with keyword-based search in a
    single query. The alpha parameter controls the weighting between
    vector (1.0) and BM25 (0.0) components.

Search Strategy:
    1. Generate dense query embedding via API
    2. Execute hybrid search in Weaviate (dense vector + BM25)
    3. Optionally generate RAG answer

Example:
    >>> pipeline = WeaviateCostOptimizedRAGSearchPipeline("config.yaml")
    >>> result = pipeline.search(
    ...     "renewable energy sources", top_k=10, filters={"category": "environment"}
    ... )
    >>> for doc in result["documents"]:
    ...     print(doc["text"][:200])
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class WeaviateCostOptimizedRAGSearchPipeline:
    """Weaviate search pipeline for cost-optimized RAG using LangChain.

    This pipeline implements hybrid search for Weaviate vector database,
    leveraging Weaviate's native hybrid search that combines dense semantic
    vectors with BM25 lexical search in a single query.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        db: WeaviateVectorDB client instance.
        collection_name: Name of the Weaviate collection.
        llm: Optional language model for RAG generation.
        search_config: Search-specific configuration.
        alpha: Weaviate hybrid weight parameter (1.0 = vector only, 0.0 = BM25 only).

    Note:
        This pipeline uses Weaviate's native hybrid search instead of
        manual RRF fusion for better performance and simplicity.

    Example:
        >>> pipeline = WeaviateCostOptimizedRAGSearchPipeline("config.yaml")
        >>> result = pipeline.search("artificial intelligence", top_k=5)
        >>> if "answer" in result:
        ...     print(result["answer"])
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain weaviate section with url and collection_name.
                Must contain embedding section with provider and model.
                Optionally contains llm section for RAG generation.
                Optionally contains search section with rrf_k and alpha.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.
        """
        # Load and validate configuration
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        # Initialize embedders
        self.dense_embedder = EmbedderHelper.create_embedder(self.config)

        # Configure Weaviate connection
        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name")

        # Initialize optional LLM
        self.llm = RAGHelper.create_llm(self.config)

        # Configure search parameters
        self.search_config = self.config.get("search", {})
        self.alpha = self.search_config.get("alpha", 0.5)  # Weaviate hybrid parameter

        logger.info(
            "Initialized Weaviate cost-optimized RAG search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute cost-optimized RAG search with Weaviate hybrid search.

        Performs hybrid search combining dense semantic vectors with BM25
        lexical search using Weaviate's native hybrid search capability.
        Optionally generates a RAG answer if LLM is configured.

        Args:
            query: Search query text.
            top_k: Number of results to return (default: 10).
            filters: Optional metadata filters as dictionary.

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents with metadata
            - query: Original search query
            - answer: Generated RAG answer (if LLM configured)

        Raises:
            RuntimeError: If search fails due to API or database errors.
        """
        # Select the collection for operations
        self.db._select_collection(self.collection_name)
        logger.info("Selected collection: %s", self.collection_name)

        # Execute hybrid search (dense vector + BM25 lexical search)
        # Weaviate provides native hybrid search with BM25 keyword matching
        merged_documents = self.db.hybrid_search(
            query=query,
            vector=EmbedderHelper.embed_query(self.dense_embedder, query),
            top_k=top_k,
            alpha=self.alpha,
            filters=filters,
            include_vectors=False,
        )
        logger.info(
            "Executed hybrid search, retrieved %d documents", len(merged_documents)
        )

        # Prepare result
        result = {
            "documents": merged_documents[:top_k],
            "query": query,
        }

        # Generate RAG answer if LLM configured
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, merged_documents[:top_k])
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
