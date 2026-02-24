"""Base indexing pipeline for query enhancement feature."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.query_enhancement.utils.config import (
    load_config,
    validate_config,
)
from vectordb.haystack.query_enhancement.utils.embeddings import (
    create_document_embedder,
)
from vectordb.utils.logging import LoggerFactory


if TYPE_CHECKING:
    from vectordb.databases.base import VectorDatabase


class BaseQueryEnhancementIndexingPipeline(ABC):
    """Abstract base class for query enhancement indexing pipelines.

    Encapsulates shared logic for loading config, dataloader, embedder,
    and executing the indexing run method. Subclasses only need to
    implement the _init_db method for database-specific initialization.

    Attributes:
        config: Configuration dictionary loaded from YAML file.
        logger: Logger instance for the pipeline.
        _dataset: Loaded dataset from configured dataloader.
        embedder: Document embedder instance.
        db: Vector database instance.
    """

    def __init__(self, config_path: str | Path, logger_name: str) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
            logger_name: Logger name (e.g., "chroma_query_enhancement_indexing").
        """
        self.config = load_config(config_path)
        validate_config(self.config)

        logger_factory = LoggerFactory(logger_name)
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

        self.logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def _init_db(self) -> "VectorDatabase":
        """Initialize the vector database from config.

        Returns:
            Initialized VectorDatabase instance.
        """
        pass

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

        # Upsert to database (subclasses can override for custom behavior)
        count = self.db.upsert(docs_with_embeddings)

        self.logger.info(f"Indexed {count} documents")
        return {"documents_indexed": count}
