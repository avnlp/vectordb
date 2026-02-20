"""Milvus hybrid search pipeline.

This module provides hybrid search capabilities for Milvus vector database,
combining dense semantic search with sparse lexical search using Milvus's
native hybrid query support and RRF ranking.

Milvus Hybrid Search:
    Milvus provides native hybrid search with automatic result fusion using
    RRF (Reciprocal Rank Fusion) or other configurable ranking strategies.
    This eliminates the need for manual result merging.

    RRF Formula:
        score = sum(1.0 / (k + rank)) for each result across all retrievers

    Where k is a constant (typically 60) that dampens the impact of low ranks.
    RRF effectively combines rankings without requiring score normalization.

Query Embedding Process:
    The search pipeline:
    1. Embeds query with dense embedder for semantic similarity
    2. Embeds query with sparse embedder for lexical matching (if configured)
    3. Passes both embeddings to Milvus search() method
    4. Milvus fuses results using configured ranker (default: rrf)

Sparse Vector Format:
    Milvus sparse vectors are stored as (index, value) pairs representing
    non-zero dimensions in a high-dimensional sparse space. The sparse
    embedder generates these using learned models like SPLADE.

Ranker Types:
    - "rrf": Reciprocal Rank Fusion (default, robust, no calibration needed)
    - "weighted": Linear score combination with configurable weights
    - "rrf_ranker": Milvus native RRF implementation

Example:
    >>> from vectordb.haystack.hybrid_indexing.search.milvus import (
    ...     MilvusHybridSearchPipeline,
    ... )
    >>> searcher = MilvusHybridSearchPipeline(
    ...     config_path="configs/milvus/triviaqa.yaml"
    ... )
    >>> results = searcher.run(query="What causes climate change?", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.score:.3f} | {doc.content[:100]}...")
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class MilvusHybridSearchPipeline:
    """Milvus hybrid (dense + sparse) search pipeline.

    Executes hybrid search queries using Milvus's native hybrid search with
    RRF ranking. Combines dense semantic similarity with sparse lexical
    matching for improved retrieval accuracy.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense query embeddings.
        sparse_embedder: Optional component for sparse query embeddings.
        db: MilvusVectorDB instance for search execution.
        collection_name: Name of the Milvus collection to query.
        ranker_type: Fusion strategy for combining dense/sparse results.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'milvus' section with collection settings and
                'embeddings' section. 'sparse' section enables hybrid search.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> searcher = MilvusHybridSearchPipeline("configs/milvus.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        # Initialize query embedders
        self.dense_embedder = EmbedderFactory.create_text_embedder(self.config)
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_text_embedder(
                self.config
            )

        # Configure Milvus connection
        milvus_config = self.config["milvus"]
        uri = milvus_config.get("uri", "http://localhost:19530")
        token = milvus_config.get("token", "")

        self.db = MilvusVectorDB(uri=uri, token=token)

        self.collection_name = milvus_config.get("collection_name")
        self.ranker_type = milvus_config.get("ranker_type", "rrf")

        logger.info("Initialized Milvus hybrid search pipeline at %s", uri)

    def run(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search query.

        Embeds the query using both dense and sparse embedders, then executes
        Milvus's native hybrid search with configured ranker (default RRF).

        Args:
            query: Search query text string.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters using Milvus filter expression
                syntax.

        Returns:
            Dictionary containing search results:
            - documents: List of ranked Document objects with scores
            - query: Original query string
            - db: Database identifier ("milvus")

        Raises:
            RuntimeError: If embedding or search operations fail.
        """
        logger.info("Running hybrid search: %s", query[:50])

        # Generate dense and optional sparse query embeddings
        dense_embedding, sparse_embedding = self._embed_query(query)

        # Execute hybrid search via Milvus native API with RRF ranking
        documents = self.db.search(
            query_embedding=dense_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            filter=filters,
            ranker_type=self.ranker_type,
        )

        logger.info("Retrieved %d documents", len(documents))
        return {
            "documents": documents,
            "query": query,
            "db": "milvus",
        }

    def _embed_query(self, query: str) -> tuple[list[float], Any | None]:
        """Embed query with dense and sparse embedders.

        Generates both dense semantic embedding and optional sparse lexical
        embedding. Dense captures query intent; sparse captures specific
        keywords for exact term matching.

        Args:
            query: Query text to embed.

        Returns:
            Tuple of (dense_embedding, sparse_embedding) where:
            - dense_embedding: List of floats for semantic similarity search
            - sparse_embedding: Sparse vector representation or None
        """
        # Generate dense semantic embedding
        dense_result = self.dense_embedder.run(text=query)
        dense_embedding = dense_result.get("embedding")

        # Generate sparse lexical embedding if configured
        sparse_embedding = None
        if self.sparse_embedder:
            sparse_result = self.sparse_embedder.run(text=query)
            sparse_embedding = sparse_result.get("sparse_embedding")

        logger.debug(
            "Embedded query (dense_dim=%s, sparse=%s)",
            len(dense_embedding) if dense_embedding else 0,
            sparse_embedding is not None,
        )

        return dense_embedding, sparse_embedding
