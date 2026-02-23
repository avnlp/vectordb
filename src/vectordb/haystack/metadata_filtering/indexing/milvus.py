"""Milvus metadata filtering indexing pipeline for Haystack.

Indexes documents with embeddings and metadata into Milvus vector database.
"""

import logging

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.metadata_filtering.common import (
    get_document_embedder,
    load_documents_from_config,
    load_metadata_filtering_config,
)


__all__ = ["MilvusMetadataFilteringIndexingPipeline"]

logger = logging.getLogger(__name__)


class MilvusMetadataFilteringIndexingPipeline:
    """Milvus metadata filtering indexing pipeline.

    Loads documents from configured dataset, embeds them, and indexes into
    Milvus with metadata for filtering.

    Attributes:
        config: Configuration dictionary.
        db: MilvusVectorDB instance.
    """

    def __init__(self, config_or_path: str | dict) -> None:
        """Initialize Milvus indexing pipeline from configuration.

        Args:
            config_or_path: Path to YAML config file or dict.

        Raises:
            ValueError: If config is invalid or required fields missing.
        """
        self.config = load_metadata_filtering_config(config_or_path)
        self._validate_config()
        self.db = self._init_db()
        logger.info("Initialized Milvus indexing pipeline")

    def _validate_config(self) -> None:
        """Validate that all required config sections exist."""
        required_sections = ["dataloader", "embeddings", "milvus"]
        for section in required_sections:
            if section not in self.config or not self.config[section]:
                raise ValueError(f"Missing or empty '{section}' in configuration")

    def _init_db(self) -> MilvusVectorDB:
        """Initialize Milvus connection.

        Returns:
            Initialized MilvusVectorDB instance.
        """
        milvus_config = self.config["milvus"]
        return MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            token=milvus_config.get("token", ""),
            collection_name=milvus_config.get("collection_name"),
        )

    def run(self) -> dict[str, int]:
        """Execute indexing pipeline.

        Steps:
        1. Load documents from configured dataset
        2. Initialize document embedder
        3. Embed documents
        4. Create collection in Milvus
        5. Insert documents with metadata

        Returns:
            Dict with 'documents_indexed' count.

        Raises:
            ValueError: If any step fails.
        """
        logger.info("Starting Milvus indexing pipeline")

        # 1. Load documents
        logger.info("Loading documents from dataset...")
        documents = load_documents_from_config(self.config)
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            raise ValueError("No documents loaded from dataset")

        # 2. Initialize embedder
        logger.info("Initializing document embedder...")
        embedder = get_document_embedder(self.config)

        # 3. Embed documents
        logger.info("Embedding documents...")
        embedder_result = embedder.run(documents=documents)
        embedded_documents = embedder_result["documents"]
        logger.info("Embedded %d documents", len(embedded_documents))

        # 4. Create collection
        milvus_config = self.config["milvus"]
        dimension = self.config["embeddings"].get("dimension", 384)
        collection_name = milvus_config.get("collection_name")

        logger.info("Creating Milvus collection: %s", collection_name)
        self.db.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            recreate=milvus_config.get("recreate", False),
        )

        # 5. Insert documents
        logger.info("Inserting documents to Milvus...")
        count = self.db.insert_documents(documents=embedded_documents)

        logger.info("Successfully indexed %d documents", count)
        return {"documents_indexed": count}
