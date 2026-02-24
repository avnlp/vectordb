"""Qdrant metadata filtering indexing pipeline for Haystack.

Indexes documents with embeddings and metadata into Qdrant vector database.
"""

import logging

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.metadata_filtering.common import (
    get_document_embedder,
    load_documents_from_config,
    load_metadata_filtering_config,
)


__all__ = ["QdrantMetadataFilteringIndexingPipeline"]

logger = logging.getLogger(__name__)


class QdrantMetadataFilteringIndexingPipeline:
    """Qdrant metadata filtering indexing pipeline.

    Loads documents from configured dataset, embeds them, and indexes into
    Qdrant with metadata for filtering.

    Attributes:
        config: Configuration dictionary.
        db: QdrantVectorDB instance.
    """

    def __init__(self, config_or_path: str | dict) -> None:
        """Initialize Qdrant indexing pipeline from configuration.

        Args:
            config_or_path: Path to YAML config file or dict.

        Raises:
            ValueError: If config is invalid or required fields missing.
        """
        self.config = load_metadata_filtering_config(config_or_path)
        self._validate_config()
        self.db = self._init_db()
        logger.info("Initialized Qdrant indexing pipeline")

    def _validate_config(self) -> None:
        """Validate that all required config sections exist."""
        required_sections = ["dataloader", "embeddings", "qdrant"]
        for section in required_sections:
            if section not in self.config or not self.config[section]:
                raise ValueError(f"Missing or empty '{section}' in configuration")

    def _init_db(self) -> QdrantVectorDB:
        """Initialize Qdrant connection.

        Returns:
            Initialized QdrantVectorDB instance.
        """
        qdrant_config = self.config["qdrant"]
        collection_name = qdrant_config.get("collection_name")
        return QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key", ""),
            collection_name=collection_name,
        )

    def run(self) -> dict[str, int]:
        """Execute indexing pipeline.

        Steps:
        1. Load documents from configured dataset
        2. Initialize document embedder
        3. Embed documents
        4. Create collection in Qdrant
        5. Index documents with metadata

        Returns:
            Dict with 'documents_indexed' count.

        Raises:
            ValueError: If any step fails.
        """
        logger.info("Starting Qdrant indexing pipeline")

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
        qdrant_config = self.config["qdrant"]
        dimension = self.config["embeddings"].get("dimension", 384)

        logger.info(
            "Creating Qdrant collection: %s", qdrant_config.get("collection_name")
        )
        self.db.create_collection(
            dimension=dimension,
            recreate=qdrant_config.get("recreate", False),
        )

        # 5. Index documents
        logger.info("Indexing documents to Qdrant...")
        count = self.db.index_documents(documents=embedded_documents)

        logger.info("Successfully indexed %d documents", count)
        return {"documents_indexed": count}
