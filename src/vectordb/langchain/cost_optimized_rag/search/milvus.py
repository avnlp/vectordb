"""Milvus cost-optimized RAG search pipeline for LangChain.

This module implements a cost-optimized search pipeline for Milvus that
combines dense semantic search with sparse lexical search using RRF fusion.

Cost Optimization:
    - Single dense query embedding: One API call for semantic search
    - Local sparse embedding: TF-IDF generated locally (zero cost)
    - RRF fusion: Lightweight local algorithm, no API calls
    - Optional LLM: RAG generation can be disabled for retrieval-only

Milvus Features:
    Milvus provides high-performance vector search with native support
    for both dense and sparse vectors, enabling efficient hybrid retrieval
    at scale with GPU acceleration support.

Search Strategy:
    1. Generate dual embeddings (dense API + sparse local)
    2. Execute parallel dense and sparse searches in Milvus
    3. Merge results using Reciprocal Rank Fusion (RRF)
    4. Optionally generate RAG answer

Example:
    >>> pipeline = MilvusCostOptimizedRAGSearchPipeline("config.yaml")
    >>> result = pipeline.search(
    ...     "vector database comparison", top_k=10, filters={"category": "databases"}
    ... )
    >>> for doc in result["documents"]:
    ...     print(doc["text"][:200])
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    ResultMerger,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class MilvusCostOptimizedRAGSearchPipeline:
    """Milvus search pipeline for cost-optimized RAG using LangChain.

    This pipeline implements hybrid search for Milvus vector database,
    combining dense semantic vectors with sparse lexical search. It uses
    Reciprocal Rank Fusion (RRF) to merge results without additional
    API calls, optimizing both cost and performance.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator.
        db: MilvusVectorDB client instance.
        collection_name: Name of the Milvus collection.
        llm: Optional language model for RAG generation.
        search_config: Search-specific configuration.
        rrf_k: RRF fusion parameter controlling rank weighting.

    Cost Benefits:
        - One dense embedding API call per query
        - Zero-cost sparse embedding (local TF-IDF)
        - Zero-cost RRF fusion (local computation)
        - Optional LLM calls only when answer generation required

    Example:
        >>> pipeline = MilvusCostOptimizedRAGSearchPipeline("config.yaml")
        >>> result = pipeline.search("embedding models", top_k=5)
        >>> if "answer" in result:
        ...     print(result["answer"])
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain milvus section with uri, db_name, and
                collection_name settings.
                Must contain embedding section with provider and model.
                Optionally contains llm section for RAG generation.
                Optionally contains search section with rrf_k parameter.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.

        Note:
            LLM is optional. If not configured, search returns documents only
            without generated answers, reducing costs for retrieval-only use cases.
        """
        # Load and validate configuration for Milvus
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        # Initialize embedders for query processing
        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        # Configure Milvus vector database connection
        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            db_name=milvus_config.get("db_name", "default"),
        )

        self.collection_name = milvus_config.get("collection_name")

        # Initialize optional LLM for RAG generation
        self.llm = RAGHelper.create_llm(self.config)

        # Configure search parameters
        self.search_config = self.config.get("search", {})
        self.rrf_k = self.search_config.get("rrf_k", 60)

        logger.info("Initialized Milvus cost-optimized RAG search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute cost-optimized RAG search with hybrid fusion.

        Performs hybrid search by combining dense semantic and sparse lexical
        results using Reciprocal Rank Fusion (RRF). Optionally generates
        a RAG answer if an LLM is configured.

        Args:
            query: Search query text. Can be natural language or keyword-based.
            top_k: Number of results to return (default: 10).
            filters: Optional metadata filters as dictionary.
                Example: {"category": "ml", "date": {"$gte": "2024-01-01"}}

        Returns:
            Dictionary containing search results:
            - documents: List of retrieved documents with metadata
            - query: The original search query
            - answer: Generated RAG answer (if LLM configured)

        Raises:
            RuntimeError: If search fails due to API or database errors.

        Example:
            >>> pipeline = MilvusCostOptimizedRAGSearchPipeline(config)
            >>> result = pipeline.search(
            ...     "transformer architecture", top_k=5, filters={"topic": "nlp"}
            ... )
            >>> print(f"Found {len(result['documents'])} documents")
        """
        # Generate dual embeddings for hybrid search
        # Dense via API for semantic understanding
        dense_query_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        # Sparse locally for lexical matching (zero API cost)
        sparse_query_embedding = self.sparse_embedder.embed_query(query)
        logger.info(
            "Embedded query with both dense and sparse embeddings: %s", query[:50]
        )

        # Execute dense semantic search
        dense_documents = self.db.query(
            vector=dense_query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            filter=filters,
        )
        logger.info("Retrieved %d documents from dense search", len(dense_documents))

        # Execute sparse lexical search
        sparse_documents = self.db.query_with_sparse(
            vector=dense_query_embedding,
            sparse_vector=sparse_query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            filter=filters,
        )
        logger.info("Retrieved %d documents from sparse search", len(sparse_documents))

        # Fuse results using Reciprocal Rank Fusion (RRF)
        # This merges dense and sparse results without additional API calls
        merged_documents = ResultMerger.merge_and_deduplicate(
            [dense_documents, sparse_documents],
            method="rrf",
            weights=[0.5, 0.5],
        )
        logger.info(
            "Fused %d unique documents from both searches", len(merged_documents)
        )

        # Prepare result with top-k merged documents
        result = {
            "documents": merged_documents[:top_k],
            "query": query,
        }

        # Generate RAG answer if LLM is configured (optional cost)
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, merged_documents[:top_k])
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
