"""Weaviate hybrid search pipeline.

This module provides hybrid search capabilities for Weaviate vector database,
combining dense semantic search with Weaviate's native BM25 lexical search.

Weaviate Hybrid Search:
    Weaviate implements hybrid search differently from other databases. Instead
    of requiring explicit sparse embeddings, Weaviate automatically:
    1. Performs dense vector search using provided embeddings
    2. Performs BM25 keyword search using internal text tokenization
    3. Fuses results with configurable alpha weighting

    This eliminates the need for sparse embedders while still providing
    hybrid retrieval capabilities.

BM25 + Vector Fusion:
    Weaviate combines BM25 and vector scores using:

        hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score

    Where alpha (0.0-1.0) controls the balance. Default alpha=0.5 gives
    equal weight to both retrieval methods.

    BM25 Advantages:
    - No embedding computation required for lexical search
    - Automatic text tokenization and indexing
    - Battle-tested TF-IDF based relevance scoring
    - Handles exact keyword matches that dense embeddings might miss

Query Processing:
    Unlike other pipelines, this pipeline only requires dense query
    embeddings. Weaviate handles the BM25 component internally using the
    raw query text.

    Process:
    1. Generate dense embedding for query (semantic component)
    2. Pass both query text and dense embedding to Weaviate
    3. Weaviate executes BM25 search internally
    4. Weaviate fuses and ranks results

No Sparse Embedder Required:
    The pipeline does not use sparse_embedder even if configured in YAML.
    This is by design - Weaviate's BM25 replaces learned sparse models.

Example:
    >>> from vectordb.haystack.hybrid_indexing.search.weaviate import (
    ...     WeaviateHybridSearchPipeline,
    ... )
    >>> searcher = WeaviateHybridSearchPipeline(
    ...     config_path="configs/weaviate/triviaqa.yaml"
    ... )
    >>> results = searcher.run(query="Deep learning fundamentals", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.score:.3f} | {doc.content[:100]}...")
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class WeaviateHybridSearchPipeline:
    """Weaviate hybrid search pipeline.

    Uses Weaviate's native hybrid search combining BM25 and vector search.
    No sparse embedder needed - Weaviate handles BM25 internally from the
    indexed text content.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense query embeddings.
        db: WeaviateVectorDB instance for search execution.
        collection_name: Name of the Weaviate collection to query.
        alpha: Weight for vector vs BM25 scores (0.0-1.0, default 0.5).
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'weaviate' section with collection settings and
                'embeddings' section for dense embedding configuration.
                Sparse section is ignored - Weaviate uses BM25 instead.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> searcher = WeaviateHybridSearchPipeline("configs/weaviate.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        # Only dense embedder needed - Weaviate handles BM25 internally
        self.dense_embedder = EmbedderFactory.create_text_embedder(self.config)

        # Configure Weaviate connection
        weaviate_config = self.config["weaviate"]
        url = weaviate_config.get("url", "http://localhost:8080")
        api_key = weaviate_config.get("api_key")

        config_dict = {"weaviate": {"url": url, "api_key": api_key}}
        self.db = WeaviateVectorDB(config=config_dict)

        self.collection_name = weaviate_config.get("collection_name")
        self.alpha = weaviate_config.get("alpha", 0.5)

        logger.info("Initialized Weaviate hybrid search pipeline at %s", url)

    def run(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search query.

        Embeds the query with dense embedder, then executes Weaviate's native
        hybrid search. Weaviate automatically performs BM25 search from the
        query text and fuses with vector search results.

        Args:
            query: Search query text string.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters for Weaviate query.

        Returns:
            Dictionary containing search results:
            - documents: List of ranked Document objects with hybrid scores
            - query: Original query string
            - db: Database identifier ("weaviate")

        Raises:
            RuntimeError: If embedding or search operations fail.
        """
        logger.info("Running hybrid search: %s", query[:50])

        # Generate dense embedding only - BM25 handled by Weaviate internally
        dense_result = self.dense_embedder.run(text=query)
        dense_embedding = dense_result.get("embedding")

        # Execute hybrid search via Weaviate wrapper (BM25 + vector fusion)
        documents = self.db.hybrid_search(
            query=query,
            query_embedding=dense_embedding,
            collection_name=self.collection_name,
            top_k=top_k,
            alpha=self.alpha,
            filter=filters,
        )

        logger.info("Retrieved %d documents", len(documents))
        return {
            "documents": documents,
            "query": query,
            "db": "weaviate",
        }
