"""Qdrant indexing pipeline for query enhancement feature."""

from pathlib import Path
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.query_enhancement.utils.config import (
    load_config,
    validate_config,
)
from vectordb.haystack.query_enhancement.utils.embeddings import (
    create_document_embedder,
)
from vectordb.utils.logging import LoggerFactory


class QdrantQueryEnhancementIndexingPipeline:
    """Index documents into Qdrant for query enhancement retrieval.

    Loads documents from configured dataloader, embeds them,
    and upserts to Qdrant collection.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        validate_config(self.config)

        logger_factory = LoggerFactory("qdrant_query_enhancement_indexing")
        self.logger = logger_factory.get_logger()

        dataloader_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dataloader_config.get("type", "triviaqa"),
            split=dataloader_config.get("split", "test"),
            limit=dataloader_config.get("limit"),
            dataset_id=dataloader_config.get("dataset_name"),
        )
        self._dataset = loader.load()
        self.embedder = create_document_embedder(self.config)
        self.db = self._init_db()

        self.logger.info("Qdrant indexing pipeline initialized")

    def _init_db(self) -> QdrantVectorDB:
        """Initialize Qdrant VectorDB from config."""
        qdrant_config = self.config.get("qdrant", {})
        return QdrantVectorDB(
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name"),
            config=self.config,
        )

    def run(self) -> dict[str, Any]:
        """Execute the indexing pipeline.

        Returns:
            Dictionary with indexing statistics.
        """
        self.logger.info("Starting document indexing")

        documents = self._dataset.to_haystack()
        self.logger.info(f"Loaded {len(documents)} documents")

        # Embed documents
        embedded = self.embedder.run(documents=documents)
        docs_with_embeddings = embedded["documents"]
        self.logger.info(f"Embedded {len(docs_with_embeddings)} documents")

        if docs_with_embeddings and docs_with_embeddings[0].embedding:
            dimension = len(docs_with_embeddings[0].embedding)
            self.db.create_index(dimension=dimension)

        # Upsert to Qdrant
        count = self.db.upsert(docs_with_embeddings)

        self.logger.info(f"Indexed {count} documents")
        return {"documents_indexed": count}
