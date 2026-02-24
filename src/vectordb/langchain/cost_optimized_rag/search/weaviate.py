"""Weaviate cost-optimized RAG search pipeline for LangChain.

This module implements a cost-optimized search pipeline for Weaviate that
combines dense semantic search with sparse lexical search using RRF fusion.

Cost Optimization:
    - Single dense query embedding: One API call for semantic search
    - Local sparse embedding: TF-IDF generated locally (zero cost)
    - RRF fusion: Local algorithm for merging results (zero cost)
    - Weaviate BM25: Native lexical search support
    - Optional LLM: RAG generation can be disabled

Weaviate Features:
    Weaviate provides native hybrid search with BM25 lexical matching,
    but this pipeline uses explicit RRF fusion for consistent behavior
    across all vector databases.

Search Strategy:
    1. Generate dual embeddings (dense API + sparse local)
    2. Execute dense vector search in Weaviate
    3. Execute sparse BM25 search in Weaviate
    4. Merge results using RRF
    5. Optionally generate RAG answer

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
    ResultMerger,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class WeaviateCostOptimizedRAGSearchPipeline:
    """Weaviate search pipeline for cost-optimized RAG using LangChain.

    This pipeline implements hybrid search for Weaviate vector database,
    combining dense semantic vectors with sparse lexical (BM25) search.
    Results are fused using Reciprocal Rank Fusion for optimal quality.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator.
        db: WeaviateVectorDB client instance.
        collection_name: Name of the Weaviate collection.
        llm: Optional language model for RAG generation.
        search_config: Search-specific configuration.
        rrf_k: RRF fusion parameter.
        alpha: Weaviate hybrid weight parameter.

    Note:
        While Weaviate has native hybrid search, this pipeline uses
        explicit RRF fusion for consistent cross-database behavior.

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
        self.sparse_embedder = SparseEmbedder()

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
        self.rrf_k = self.search_config.get("rrf_k", 60)
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
        """Execute cost-optimized RAG search with hybrid fusion.

        Performs hybrid search by combining dense semantic and sparse lexical
        results using RRF. Optionally generates a RAG answer if LLM is configured.

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
        # Generate dual embeddings for hybrid search
        dense_query_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        sparse_query_embedding = self.sparse_embedder.embed_query(query)
        logger.info(
            "Embedded query with both dense and sparse embeddings: %s", query[:50]
        )

        # Execute dense vector search
        dense_documents = self.db.query(
            vector=dense_query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            where=filters,
        )
        logger.info("Retrieved %d documents from dense search", len(dense_documents))

        # Execute sparse BM25 search
        sparse_documents = self.db.query_with_sparse(
            vector=dense_query_embedding,
            sparse_vector=sparse_query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            where=filters,
        )
        logger.info("Retrieved %d documents from sparse search", len(sparse_documents))

        # Fuse results using RRF
        merged_documents = ResultMerger.merge_and_deduplicate(
            [dense_documents, sparse_documents],
            method="rrf",
            weights=[0.5, 0.5],
        )
        logger.info(
            "Fused %d unique documents from both searches", len(merged_documents)
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
