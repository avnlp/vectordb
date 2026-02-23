"""Milvus reranking search pipeline.

This module provides a search pipeline that combines Milvus vector similarity
search with cross-encoder reranking for improved precision. Milvus's high-
performance distributed architecture enables scalable candidate retrieval.

Reranking Strategy:
    1. Embed query using bi-encoder for dense vector representation
    2. Retrieve 3x candidates from Milvus using GPU-accelerated ANN search
    3. Score candidates with cross-encoder for precise relevance
    4. Return top_k results ordered by cross-encoder score

Milvus Capabilities:
    - Distributed vector database for billion-scale collections
    - Multiple index types: IVF, HNSW, DiskANN for different trade-offs
    - GPU acceleration for large-scale similarity search
    - Rich attribute filtering and hybrid search

Two-Stage Benefits:
    Bi-encoder retrieval handles scale (millions of documents), while
    cross-encoder reranking handles precision (final top_k selection).
    This division of labor optimizes the cost-quality trade-off.
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory, RerankerFactory


logger = logging.getLogger(__name__)


class MilvusRerankingSearchPipeline:
    """Milvus search pipeline with cross-encoder reranking.

    Implements two-stage retrieval using Milvus for scalable vector search
    and cross-encoder for precise final ranking. Suitable for large-scale
    deployments requiring high-precision results.

    Attributes:
        config: Pipeline configuration with milvus, embedder, reranker sections.
        embedder: Bi-encoder for query embedding and document retrieval.
        reranker: Cross-encoder for final document scoring.
        db: MilvusVectorDB instance for distributed vector storage.
        collection_name: Name of the Milvus collection being searched.

    Example:
        >>> pipeline = MilvusRerankingSearchPipeline("milvus_config.yaml")
        >>> results = pipeline.search(
        ...     query="large language model training",
        ...     top_k=10,
        ...     filters={"year": {"$gte": 2023}},
        ... )
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - milvus: uri, collection_name, token (if using Zilliz Cloud)
                - embedder: Provider, model, dimensions for bi-encoder
                - reranker: Provider, model for cross-encoder

        Raises:
            ValueError: If required config sections are missing or invalid.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        self.reranker = RerankerFactory.create(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            collection_name=milvus_config.get("collection_name", "reranking"),
        )

        self.collection_name = milvus_config.get("collection_name", "reranking")

        logger.info("Initialized Milvus reranking search pipeline")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search with two-stage retrieval.

        Stage 1 - Dense Retrieval:
            Query is embedded and Milvus retrieves 3x candidates using
            vector similarity with HNSW or IVF index.

        Stage 2 - Cross-Encoder Reranking:
            Cross-encoder scores each candidate document against the query
            for precise relevance assessment, returning only top_k results.

        Args:
            query: Search query text to find relevant documents for.
            top_k: Number of final results to return after reranking.
            filters: Optional metadata filters to apply during search.
                Supports Milvus's expression syntax for complex conditions.

        Returns:
            Dict with 'query' and 'documents' keys. Documents are ordered by
            cross-encoder relevance score (highest first).
        """
        # Stage 1: Embed query using bi-encoder for vector search
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Retrieve 3x candidates to maximize coverage for reranking stage
        retrieval_top_k = top_k * 3
        base_docs = self.db.search(
            query_embedding=query_embedding,
            top_k=retrieval_top_k,
            filters=filters,
        )
        logger.info("Retrieved %d base documents", len(base_docs))

        if not base_docs:
            logger.warning("No base documents retrieved")
            return {"query": query, "documents": []}

        # Stage 2: Apply cross-encoder reranking for precision improvement
        # Cross-encoder captures query-document interactions missed by bi-encoders
        reranked_result = self.reranker.run(query=query, documents=base_docs)
        reranked_docs = reranked_result.get("documents", [])[:top_k]

        logger.info("Reranked to %d documents", len(reranked_docs))
        return {"query": query, "documents": reranked_docs}

    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Alias for search() for backward compatibility with Haystack pipelines."""
        return self.search(query, top_k)
