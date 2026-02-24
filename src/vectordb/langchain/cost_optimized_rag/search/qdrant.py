"""Qdrant cost-optimized RAG search pipeline for LangChain.

This module implements a cost-optimized search pipeline for Qdrant that
combines dense semantic search with sparse lexical search using RRF fusion.

Cost Optimization:
    - Single dense query embedding: One API call for semantic search
    - Local sparse embedding: TF-IDF generated locally (zero cost)
    - Native hybrid search: Single API call with RRF fusion done server-side
    - Optional LLM: RAG generation can be disabled for retrieval-only

Qdrant Features:
    Qdrant provides native support for both dense and sparse vectors,
    enabling efficient hybrid search with optimized storage.

Search Strategy:
    1. Generate dual embeddings (dense API + sparse local)
    2. Execute native hybrid search (dense + sparse + RRF in one call)
    3. Optionally generate RAG answer

Example:
    >>> pipeline = QdrantCostOptimizedRAGSearchPipeline("config.yaml")
    >>> result = pipeline.search(
    ...     "cloud computing benefits", top_k=10, filters={"category": "infrastructure"}
    ... )
    >>> for doc in result["documents"]:
    ...     print(doc["payload"]["text"][:200])
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class QdrantCostOptimizedRAGSearchPipeline:
    """Qdrant search pipeline for cost-optimized RAG using LangChain.

    This pipeline implements hybrid search for Qdrant vector database,
    leveraging its native support for both dense and sparse vectors.
    Results are merged using Reciprocal Rank Fusion for optimal
    retrieval quality at minimal cost.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator.
        db: QdrantVectorDB client instance.
        collection_name: Name of the Qdrant collection.
        llm: Optional language model for RAG generation.
        search_config: Search-specific configuration.
        rrf_k: RRF fusion parameter.

    Cost Benefits:
        - One API call for dense query embedding
        - Zero-cost sparse query embedding
        - Zero-cost RRF fusion
        - Optional LLM only when answer needed

    Example:
        >>> pipeline = QdrantCostOptimizedRAGSearchPipeline("config.yaml")
        >>> result = pipeline.search("machine learning", top_k=5)
        >>> print(f"Retrieved {len(result['documents'])} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Qdrant search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain qdrant section with url and collection_name.
                Must contain embedding section with provider and model.
                Optionally contains llm section for RAG generation.
                Optionally contains search section with rrf_k parameter.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.
        """
        # Load and validate configuration
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        # Initialize embedders
        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        # Configure Qdrant connection
        self.db = QdrantVectorDB(config=self.config)
        self.collection_name = self.config["qdrant"]["collection_name"]

        # Initialize optional LLM
        self.llm = RAGHelper.create_llm(self.config)

        # Configure search parameters (rrf_k kept for config compatibility, not used)
        self.search_config = self.config.get("search", {})
        self.rrf_k = self.search_config.get("rrf_k", 60)

        logger.info("Initialized Qdrant cost-optimized RAG search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute cost-optimized RAG search with hybrid fusion.

        Performs hybrid search combining dense semantic and sparse lexical
        results using RRF. Optionally generates a RAG answer.

        Args:
            query: Search query text.
            top_k: Number of results to return (default: 10).
            filters: Optional metadata filters as dictionary.

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents
            - query: Original search query
            - answer: Generated RAG answer (if LLM configured)

        Raises:
            RuntimeError: If search fails due to API or database errors.
        """
        # Generate dual embeddings for hybrid search
        dense_query_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        sparse_query_embedding = self.sparse_embedder.embed_query(query)
        logger.info(
            "Embedded query with both dense and sparse embeddings: %s", query[:50]
        )

        # Use native hybrid search - single API call with RRF fusion done internally
        documents = self.db.search(
            query_vector={
                "dense": dense_query_embedding,
                "sparse": sparse_query_embedding,
            },
            search_type="hybrid",
            scope=self.collection_name,
            top_k=top_k,
            filters=filters,
        )
        logger.info("Retrieved %d documents from hybrid search", len(documents))

        # Prepare result
        result = {
            "documents": documents,
            "query": query,
        }

        # Generate RAG answer if LLM configured
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
