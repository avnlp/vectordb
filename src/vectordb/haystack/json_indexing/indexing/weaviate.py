"""Weaviate JSON indexing pipeline for Haystack."""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.json_indexing.common.config import load_config
from vectordb.haystack.json_indexing.common.embeddings import (
    create_document_embedder,
    embed_documents,
)
from vectordb.utils.logging import LoggerFactory


class WeaviateJSONIndexer:
    """Indexes documents into Weaviate with JSON metadata.

    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        vector_db: WeaviateVectorDB wrapper instance.
        embedder: Document embedder.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexer from configuration.

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
        name = logging_config.get("name", "weaviate_json_indexing")
        level_str = logging_config.get("level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        factory = LoggerFactory(name, log_level=level)
        self.logger = factory.get_logger()

    def _connect(self) -> None:
        """Connect to Weaviate using VectorDB wrapper."""
        weaviate_config = self.config.get("weaviate", {})
        cluster_url = weaviate_config.get("cluster_url")
        api_key = weaviate_config.get("api_key")

        self.vector_db = WeaviateVectorDB(cluster_url=cluster_url, api_key=api_key)
        self.logger.info("Connected to Weaviate")

    def _init_embedder(self) -> None:
        """Initialize document embedder."""
        self.embedder = create_document_embedder(self.config)
        self.logger.info("Initialized document embedder")

    def run(self) -> dict[str, Any]:
        """Execute full indexing pipeline.

        Returns:
            Dictionary with indexing statistics.
        """
        self.logger.info("Starting JSON indexing pipeline")

        dataloader_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dataloader_config.get("type", "triviaqa"),
            split=dataloader_config.get("split", "test"),
            limit=dataloader_config.get("limit"),
            dataset_id=dataloader_config.get("dataset_name"),
        )
        documents = loader.load().to_haystack()

        # Apply limit
        limit = dataloader_config.get("limit")
        if limit:
            self.logger.info("Limited to %d documents", limit)

        self.logger.info("Loaded %d documents", len(documents))

        collection_config = self.config.get("collection", {})
        collection_name = collection_config.get("name", "json_indexed")

        self.vector_db.create_collection(
            collection_name=collection_name,
            skip_vectorization=False,
        )
        self.logger.info("Created collection: %s", collection_name)

        # Embed and index
        embedded_docs = embed_documents(documents, self.embedder)
        self.vector_db.upsert_documents(embedded_docs, collection_name)
        self.logger.info("Indexed %d documents", len(embedded_docs))

        return {"documents_indexed": len(documents)}
