"""Milvus JSON search pipeline for Haystack."""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.json_indexing.common.config import load_config
from vectordb.haystack.json_indexing.common.embeddings import create_text_embedder
from vectordb.haystack.json_indexing.common.filters.milvus import build_milvus_filter
from vectordb.utils.logging import LoggerFactory


class MilvusJSONSearcher:
    """Searches Milvus with JSON metadata filtering.

    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        vector_db: MilvusVectorDB wrapper instance.
        text_embedder: Text embedder for queries.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize searcher from configuration.

        Args:
            config_or_path: Path to YAML file or config dict.
        """
        self.config = load_config(config_or_path)
        self._setup_logging()
        self._connect()
        self._init_embedder()

    def _setup_logging(self) -> None:
        """Set up logger from config."""
        logging_config = self.config.get("logging", {})
        name = logging_config.get("name", "milvus_json_search")
        level_str = logging_config.get("level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        factory = LoggerFactory(name, log_level=level)
        self.logger = factory.get_logger()

    def _connect(self) -> None:
        """Connect to Milvus using VectorDB wrapper."""
        milvus_config = self.config.get("milvus", {})
        self.vector_db = MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            token=milvus_config.get("token", ""),
        )
        self.logger.info("Connected to Milvus")

    def _init_embedder(self) -> None:
        """Initialize text embedder."""
        self.text_embedder = create_text_embedder(self.config)
        self.logger.info("Initialized text embedder")

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search with optional JSON filters.

        Args:
            query: Search query text.
            filters: Optional filter conditions.
            top_k: Number of results to return.

        Returns:
            List of search results with content and metadata.
        """
        if top_k is None:
            top_k = self.config.get("search", {}).get("top_k", 10)

        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "json_indexed")
        indexing_config = self.config.get("indexing", {})
        json_field_name = indexing_config.get("json_field_name", "metadata")

        # Embed query
        result = self.text_embedder.run(text=query)
        query_embedding = result["embedding"]

        filter_expr = build_milvus_filter(filters, json_field_name)

        # Execute search via wrapper
        results = self.vector_db.search(
            query_embedding=query_embedding,
            collection_name=collection_name,
            limit=top_k,
            filter_expr=filter_expr,
        )

        self.logger.info(
            "Search for '%s' returned %d results", query[:50], len(results)
        )
        return results
