"""Pinecone metadata filtering indexing pipeline for Haystack.

Indexes documents with embeddings and metadata into Pinecone vector database.
"""

import logging

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.metadata_filtering.common import (
    get_document_embedder,
    load_documents_from_config,
    load_metadata_filtering_config,
)


__all__ = ["PineconeMetadataFilteringIndexingPipeline"]

logger = logging.getLogger(__name__)


class PineconeMetadataFilteringIndexingPipeline:
    """Pinecone metadata filtering indexing pipeline.

    Loads documents from configured dataset, embeds them, and indexes into
    Pinecone with metadata for filtering.

    Attributes:
        config: Configuration dictionary.
        db: PineconeVectorDB instance.
    """

    def __init__(self, config_or_path: str | dict) -> None:
        """Initialize Pinecone indexing pipeline from configuration.

        Args:
            config_or_path: Path to YAML config file or dict.

        Raises:
            ValueError: If config is invalid or required fields missing.
        """
        self.config = load_metadata_filtering_config(config_or_path)
        self._validate_config()
        self.db = PineconeVectorDB(config=self.config)
        logger.info("Initialized Pinecone indexing pipeline")

    def _validate_config(self) -> None:
        """Validate that all required config sections exist."""
        required_sections = ["dataloader", "embeddings", "pinecone"]
        for section in required_sections:
            if section not in self.config or not self.config[section]:
                raise ValueError(f"Missing or empty '{section}' in configuration")

    def run(self) -> dict[str, int]:
        """Execute indexing pipeline.

        Steps:
        1. Load documents from configured dataset
        2. Initialize document embedder
        3. Embed documents
        4. Create index in Pinecone
        5. Upsert documents with metadata

        Returns:
            Dict with 'documents_indexed' count.

        Raises:
            ValueError: If any step fails.
        """
        logger.info("Starting Pinecone indexing pipeline")

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

        # 4. Create index
        pinecone_config = self.config["pinecone"]
        dimension = self.config["embeddings"].get("dimension", 384)

        logger.info("Creating Pinecone index...")
        self.db.create_index(
            index_name=pinecone_config["index_name"],
            dimension=dimension,
            metric=pinecone_config.get("metric", "cosine"),
        )

        # 5. Upsert documents
        logger.info("Upserting documents to Pinecone...")
        count = self.db.upsert(
            data=embedded_documents,
            namespace=pinecone_config.get("namespace", "default"),
        )

        logger.info("Successfully indexed %d documents", count)
        return {"documents_indexed": count}
