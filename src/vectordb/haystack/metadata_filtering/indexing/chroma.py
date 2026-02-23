"""Chroma metadata filtering indexing pipeline for Haystack.

Indexes documents with embeddings and metadata into Chroma vector database.
"""

import logging

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.haystack.metadata_filtering.common import (
    get_document_embedder,
    load_documents_from_config,
    load_metadata_filtering_config,
)


__all__ = ["ChromaMetadataFilteringIndexingPipeline"]

logger = logging.getLogger(__name__)


class ChromaMetadataFilteringIndexingPipeline:
    """Chroma metadata filtering indexing pipeline.

    Loads documents from configured dataset, embeds them, and indexes into
    Chroma with metadata for filtering.

    Attributes:
        config: Configuration dictionary.
        db: ChromaVectorDB instance.
    """

    def __init__(self, config_or_path: str | dict) -> None:
        """Initialize Chroma indexing pipeline from configuration.

        Args:
            config_or_path: Path to YAML config file or dict.

        Raises:
            ValueError: If config is invalid or required fields missing.
        """
        self.config = load_metadata_filtering_config(config_or_path)
        self._validate_config()
        self.db = self._init_db()
        logger.info("Initialized Chroma indexing pipeline")

    def _validate_config(self) -> None:
        """Validate that all required config sections exist."""
        required_sections = ["dataloader", "embeddings", "chroma"]
        for section in required_sections:
            if section not in self.config or not self.config[section]:
                raise ValueError(f"Missing or empty '{section}' in configuration")

    def _init_db(self) -> ChromaVectorDB:
        """Initialize Chroma connection.

        Returns:
            Initialized ChromaVectorDB instance.
        """
        chroma_config = self.config["chroma"]
        return ChromaVectorDB(
            persist_directory=chroma_config.get("persist_directory", "./chroma_data"),
            collection_name=chroma_config.get("collection_name"),
        )

    def run(self) -> dict[str, int]:
        """Execute indexing pipeline.

        Steps:
        1. Load documents from configured dataset
        2. Initialize document embedder
        3. Embed documents
        4. Create collection in Chroma
        5. Add documents with metadata

        Returns:
            Dict with 'documents_indexed' count.

        Raises:
            ValueError: If any step fails.
        """
        logger.info("Starting Chroma indexing pipeline")

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
        chroma_config = self.config["chroma"]
        collection_name = chroma_config.get("collection_name")

        logger.info("Creating Chroma collection: %s", collection_name)
        self.db.create_collection(
            collection_name=collection_name,
            recreate=chroma_config.get("recreate", False),
        )

        # 5. Add documents
        logger.info("Adding documents to Chroma...")
        count = self.db.add_documents(documents=embedded_documents)

        logger.info("Successfully indexed %d documents", count)
        return {"documents_indexed": count}
