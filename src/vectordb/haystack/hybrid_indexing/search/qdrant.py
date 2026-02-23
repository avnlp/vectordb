"""Qdrant hybrid search pipeline.

This module provides hybrid search capabilities for Qdrant vector database,
combining dense semantic search with sparse lexical search using Qdrant's
native hybrid query API.

Qdrant Hybrid Search:
    Qdrant's hybrid search performs both dense and sparse vector lookups,
    then fuses results using internal ranking algorithms. This provides
    a unified interface for hybrid retrieval without manual fusion code.

    Key Features:
    - Native hybrid query type combining dense and sparse vectors
    - Automatic result fusion and ranking
    - Support for query-time fusion parameters
    - Consistent API across dense, sparse, and hybrid searches

Query Embedding Process:
    The pipeline generates dual representations:
    1. Dense embedding: Semantic vector from sentence transformer
    2. Sparse embedding: Lexical vector from SPLADE or similar model

    Both are passed to Qdrant's search() with search_type="hybrid".

Sparse Vector Format:
    Qdrant sparse vectors are dictionaries mapping dimension indices to
    float values. Only non-zero entries are stored, making them efficient
    for high-dimensional learned sparse representations.

Search Type Parameter:
    Qdrant supports multiple search types:
    - "dense": Semantic similarity only
    - "sparse": Lexical matching only
    - "hybrid": Combined dense + sparse with automatic fusion

Example:
    >>> from vectordb.haystack.hybrid_indexing.search.qdrant import (
    ...     QdrantHybridSearchPipeline,
    ... )
    >>> searcher = QdrantHybridSearchPipeline(
    ...     config_path="configs/qdrant/triviaqa.yaml"
    ... )
    >>> results = searcher.run(query="Quantum computing applications", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.score:.3f} | {doc.content[:100]}...")
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class QdrantHybridSearchPipeline:
    """Qdrant hybrid (dense + sparse) search pipeline.

    Executes hybrid search queries using Qdrant's native hybrid search
    capabilities. Combines dense semantic similarity with sparse lexical
    matching through Qdrant's unified hybrid query interface.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense query embeddings.
        sparse_embedder: Optional component for sparse query embeddings.
        db: QdrantVectorDB instance for search execution.
        collection_name: Name of the Qdrant collection to query.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'qdrant' section with collection settings and
                'embeddings' section. 'sparse' section enables hybrid search.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> searcher = QdrantHybridSearchPipeline("configs/qdrant.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        # Initialize query embedders
        self.dense_embedder = EmbedderFactory.create_text_embedder(self.config)
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_text_embedder(
                self.config
            )

        # Configure Qdrant connection
        qdrant_config = self.config["qdrant"]
        url = qdrant_config.get("url", "http://localhost:6333")
        api_key = qdrant_config.get("api_key")
        path = qdrant_config.get("path")

        config_dict = {"qdrant": {"url": url, "api_key": api_key, "path": path}}
        self.db = QdrantVectorDB(config=config_dict)

        self.collection_name = qdrant_config.get("collection_name")

        logger.info("Initialized Qdrant hybrid search pipeline at %s", url)

    def run(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search query.

        Embeds the query using both dense and sparse embedders, then executes
        Qdrant's native hybrid search with automatic result fusion.

        Args:
            query: Search query text string.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters using Qdrant filter syntax.

        Returns:
            Dictionary containing search results:
            - documents: List of ranked Document objects with scores
            - query: Original query string
            - db: Database identifier ("qdrant")

        Raises:
            RuntimeError: If embedding or search operations fail.
        """
        logger.info("Running hybrid search: %s", query[:50])

        # Generate dense and optional sparse query embeddings
        dense_embedding, sparse_embedding = self._embed_query(query)

        # Execute hybrid search via Qdrant's native API
        documents = self.db.search(
            query_vector=dense_embedding,
            query_sparse_vector=sparse_embedding,
            collection_name=self.collection_name,
            top_k=top_k,
            filter=filters,
            search_type="hybrid",
        )

        logger.info("Retrieved %d documents", len(documents))
        return {
            "documents": documents,
            "query": query,
            "db": "qdrant",
        }

    def _embed_query(self, query: str) -> tuple[list[float], Any | None]:
        """Embed query with dense and sparse embedders.

        Generates both dense semantic embedding and optional sparse lexical
        embedding for hybrid search. The dense embedding captures semantic
        meaning while sparse captures keyword matches.

        Args:
            query: Query text to embed.

        Returns:
            Tuple of (dense_embedding, sparse_embedding) where:
            - dense_embedding: List of floats for semantic search
            - sparse_embedding: Dict mapping indices to weights, or None
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
