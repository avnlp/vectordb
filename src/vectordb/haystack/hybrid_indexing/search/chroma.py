"""Chroma hybrid search pipeline.

This module provides hybrid search capabilities for Chroma vector database,
combining dense semantic search with sparse lexical search through manual
result fusion using RRF (Reciprocal Rank Fusion).

Chroma Hybrid Search:
    Unlike databases with native hybrid search, Chroma requires manual
    fusion of dense and sparse search results. This pipeline:
    1. Executes dense vector search for semantic similarity
    2. Executes sparse vector search for lexical matching (if configured)
    3. Fuses results using Reciprocal Rank Fusion (RRF)

RRF Fusion Strategy:
    Reciprocal Rank Fusion combines rankings from multiple sources without
    requiring score normalization:

        rrf_score = sum(1.0 / (k + rank)) for each result in each list

    Where k=60 is a constant that dampens the impact of low ranks. Documents
    appearing in multiple result lists get higher fused scores.

    Advantages:
    - No score calibration required between dense and sparse
    - Robust to different score distributions
    - Simple and effective ranking combination

Query Embedding Process:
    The pipeline generates dual query representations:
    1. Dense embedding: Semantic vector for similarity search
    2. Sparse embedding: Lexical vector for term matching

    Dense search always executes. Sparse search executes only if sparse
    embedder is configured and produces valid embeddings.

Fallback Behavior:
    If sparse search is not available (no embedder or no results), the
    pipeline falls back to dense-only results, returning top_k from the
    dense search.

Example:
    >>> from vectordb.haystack.hybrid_indexing.search.chroma import (
    ...     ChromaHybridSearchPipeline,
    ... )
    >>> searcher = ChromaHybridSearchPipeline(
    ...     config_path="configs/chroma/triviaqa.yaml"
    ... )
    >>> results = searcher.run(query="Neural network architectures", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.score:.3f} | {doc.content[:100]}...")
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory, ResultMerger


logger = logging.getLogger(__name__)


class ChromaHybridSearchPipeline:
    """Chroma hybrid search pipeline.

    Performs hybrid search by executing separate dense and sparse searches,
    then fusing results using Reciprocal Rank Fusion (RRF). Chroma does not
    have native hybrid search, requiring manual result combination.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense query embeddings.
        sparse_embedder: Optional component for sparse query embeddings.
        db: ChromaVectorDB instance for search execution.
        collection_name: Name of the Chroma collection to query.
        fusion_strategy: Result fusion method (default: "rrf").
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'chroma' section with collection settings and
                'embeddings' section. 'sparse' section enables hybrid search.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> searcher = ChromaHybridSearchPipeline("configs/chroma.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        # Initialize query embedders
        self.dense_embedder = EmbedderFactory.create_text_embedder(self.config)
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_text_embedder(
                self.config
            )

        # Configure Chroma connection
        chroma_config = self.config["chroma"]
        host = chroma_config.get("host", "localhost")
        port = chroma_config.get("port", 8000)
        persistent = chroma_config.get("persistent", False)
        path = chroma_config.get("path")

        config_dict = {
            "chroma": {
                "host": host,
                "port": port,
                "persistent": persistent,
                "path": path,
            }
        }
        self.db = ChromaVectorDB(config=config_dict)

        self.collection_name = chroma_config.get("collection_name")
        self.fusion_strategy = chroma_config.get("fusion_strategy", "rrf")

        logger.info("Initialized Chroma hybrid search pipeline at %s:%d", host, port)

    def run(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search query.

        Performs dense search (always) and sparse search (if configured),
        then fuses results using RRF. Falls back to dense-only if sparse
        is unavailable.

        Args:
            query: Search query text string.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters for Chroma query.

        Returns:
            Dictionary containing search results:
            - documents: List of ranked Document objects with fused scores
            - query: Original query string
            - db: Database identifier ("chroma")

        Raises:
            RuntimeError: If embedding or search operations fail.
        """
        logger.info("Running hybrid search: %s", query[:50])

        # Generate both dense and sparse query embeddings
        dense_embedding, sparse_embedding = self._embed_query(query)

        # Execute dense semantic search (always performed)
        dense_docs = self.db.search(
            query_embedding=dense_embedding,
            collection_name=self.collection_name,
            top_k=top_k * 2,
            filter=filters,
            search_type="dense",
        )
        logger.debug("Dense search retrieved %d documents", len(dense_docs))

        # Execute sparse lexical search if embedder configured
        sparse_docs = []
        if sparse_embedding and self.sparse_embedder:
            sparse_docs = self.db.search(
                query_embedding=sparse_embedding,
                collection_name=self.collection_name,
                top_k=top_k * 2,
                filter=filters,
                search_type="sparse",
            )
            logger.debug("Sparse search retrieved %d documents", len(sparse_docs))

        # Fuse results using RRF if sparse results available, else dense-only
        if sparse_docs:
            documents = ResultMerger.fuse(
                dense_docs, sparse_docs, top_k=top_k, strategy=self.fusion_strategy
            )
        else:
            # Fallback to top_k from dense-only results
            documents = dense_docs[:top_k]

        logger.info("Retrieved %d documents (fused)", len(documents))
        return {
            "documents": documents,
            "query": query,
            "db": "chroma",
        }

    def _embed_query(self, query: str) -> tuple[list[float], Any | None]:
        """Embed query with dense and sparse embedders.

        Generates dense semantic embedding and optional sparse lexical
        embedding. Both representations are used for hybrid search, with
        sparse being optional based on configuration.

        Args:
            query: Query text to embed.

        Returns:
            Tuple of (dense_embedding, sparse_embedding) where:
            - dense_embedding: List of floats for semantic similarity
            - sparse_embedding: Sparse vector dict or None if not configured
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
