"""Pinecone hybrid search pipeline.

This module provides hybrid search capabilities for Pinecone vector database,
combining dense semantic search with sparse lexical search using Pinecone's
native hybrid query API.

Pinecone Hybrid Search:
    Pinecone's native hybrid search accepts both dense and sparse query vectors,
    performing parallel searches and fusing results with configurable alpha
    weighting.

    Fusion Formula:
        final_score = alpha * dense_score + (1 - alpha) * sparse_score

    Where alpha controls the balance between semantic (dense) and lexical
    (sparse) relevance. Default alpha=0.5 gives equal weight to both.

Query Embedding Process:
    The search pipeline embeds queries using both:
    1. Dense embedder: Captures semantic meaning and intent
    2. Sparse embedder: Captures specific keywords and terms

    Both embeddings are passed to Pinecone's hybrid_search() method which
    handles the dual retrieval and score fusion internally.

Sparse Vector Format:
    Sparse embeddings are represented as dictionaries mapping token indices
    to importance weights. Pinecone expects this format for sparse_vector
    query parameters.

Example:
    >>> from vectordb.haystack.hybrid_indexing.search.pinecone import (
    ...     PineconeHybridSearchPipeline,
    ... )
    >>> searcher = PineconeHybridSearchPipeline(
    ...     config_path="configs/pinecone/triviaqa.yaml"
    ... )
    >>> results = searcher.run(query="What is machine learning?", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.score:.3f} | {doc.content[:100]}...")
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class PineconeHybridSearchPipeline:
    """Pinecone hybrid (dense + sparse) search pipeline.

    Executes hybrid search queries using Pinecone's native sparse_vector
    support. Combines dense semantic similarity with sparse lexical matching
    using configurable alpha weighting.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense query embeddings.
        sparse_embedder: Optional component for sparse query embeddings.
        db: PineconeVectorDB instance for search execution.
        index_name: Name of the Pinecone index to query.
        namespace: Namespace for document isolation.
        alpha: Weight for dense vs sparse scores (0.0-1.0, default 0.5).
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'pinecone' section with index settings and
                'embeddings' section. 'sparse' section enables hybrid search.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> searcher = PineconeHybridSearchPipeline("configs/pinecone.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        # Initialize query embedders
        self.dense_embedder = EmbedderFactory.create_text_embedder(self.config)
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_text_embedder(
                self.config
            )

        # Configure Pinecone connection
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config.get("api_key"),
            index_name=pinecone_config.get("index_name"),
            host=pinecone_config.get("host"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "default")
        self.alpha = pinecone_config.get("alpha", 0.5)

        logger.info("Initialized Pinecone hybrid search pipeline")

    def run(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search query.

        Embeds the query using both dense and sparse embedders, then executes
        Pinecone's native hybrid search with alpha-weighted score fusion.

        Args:
            query: Search query text string.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters for Pinecone query.
                Supports Pinecone's filter expression format.

        Returns:
            Dictionary containing search results:
            - documents: List of ranked Document objects with scores
            - query: Original query string
            - db: Database identifier ("pinecone")

        Raises:
            RuntimeError: If embedding or search operations fail.
        """
        logger.info("Running hybrid search: %s", query[:50])

        # Generate dense and optional sparse query embeddings
        dense_embedding, sparse_embedding = self._embed_query(query)

        # Execute hybrid search via Pinecone's native API
        documents = self.db.hybrid_search(
            query_embedding=dense_embedding,
            query_sparse_embedding=sparse_embedding,
            index_name=self.index_name,
            namespace=self.namespace,
            top_k=top_k,
            filter=filters,
            alpha=self.alpha,
        )

        logger.info("Retrieved %d documents", len(documents))
        return {
            "documents": documents,
            "query": query,
            "db": "pinecone",
        }

    def _embed_query(self, query: str) -> tuple[list[float], Any | None]:
        """Embed query with dense and sparse embedders.

        Generates both dense semantic embedding and optional sparse lexical
        embedding for the query text. Sparse embeddings capture specific
        keywords that dense embeddings might miss.

        Args:
            query: Query text to embed.

        Returns:
            Tuple of (dense_embedding, sparse_embedding) where:
            - dense_embedding: List of floats representing semantic meaning
            - sparse_embedding: Dict mapping token indices to weights, or None
              if sparse embedder not configured
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
