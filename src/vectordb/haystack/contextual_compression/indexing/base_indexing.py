"""Base class for contextual compression indexing pipelines.

Provides shared functionality for indexing documents across all vector databases.
Indexing is the prerequisite step before contextual compression search can be used.

Indexing Pipeline Flow:
    1. Load dataset from configured source (TriviaQA, ARC, PopQA, etc.)
    2. Initialize dense embedder (default: Qwen/Qwen3-Embedding-0.6B)
    3. Generate embeddings in configurable batches (default: 32)
    4. Store in vector database with content and metadata

Database Schema:
    All pipelines store: id, content (text), embedding (vector), metadata (JSON)
    Schema varies slightly by database capabilities (e.g., Milvus uses IVF_FLAT index).

Subclasses must implement:
    - _connect(): Database connection logic
    - _prepare_collection(): Collection/index creation
    - _store_documents(): Batch document storage
"""

from abc import ABC, abstractmethod
from typing import Any

from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.json_indexing.common.config import load_config
from vectordb.utils.config import setup_logger


class BaseIndexingPipeline(ABC):
    """Abstract base class for contextual compression indexing pipelines.

    Provides shared initialization, dataloader setup, embedder initialization,
    and indexing orchestration. Subclasses implement database-specific storage.

    Attributes:
        config: Configuration dictionary loaded from YAML.
        logger: Logger instance for the pipeline.
        dense_embedder: Text embedder for document embeddings.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)
        self._init_embedders()
        self._connect()
        self._prepare_collection()

    def _init_embedders(self) -> None:
        """Initialize dense embedder from configuration."""
        embeddings_config = self.config.get("embeddings", {})
        dense_model = embeddings_config.get("model", "Qwen/Qwen3-Embedding-0.6B")

        # Support model aliases
        model_aliases = {
            "qwen3": "Qwen/Qwen3-Embedding-0.6B",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        }
        dense_model = model_aliases.get(dense_model.lower(), dense_model)

        self.dense_embedder = SentenceTransformersTextEmbedder(model=dense_model)
        self.dense_embedder.warm_up()
        self.logger.info("Initialized dense embedder with model: %s", dense_model)

    @abstractmethod
    def _connect(self) -> None:
        """Establish connection to the vector database.

        Subclasses must implement database-specific connection logic.
        """

    @abstractmethod
    def _prepare_collection(self) -> None:
        """Prepare/create the database collection for storage.

        Subclasses must implement database-specific collection preparation.
        """

    @abstractmethod
    def _store_documents(self, documents: list[Document]) -> None:
        """Store embedded documents in the vector database.

        Args:
            documents: List of Document objects with embeddings.
        """

    def _load_dataset(self) -> list[Document]:
        """Load dataset from configuration and convert to Documents.

        Returns:
            List of Document objects ready for embedding.
        """
        dataset_config = self.config.get("dataset", {})
        dataset_type = dataset_config.get("type")
        dataset_name = dataset_config.get("name")
        split = dataset_config.get("split", "test")
        limit = dataset_config.get("limit")

        if not dataset_type:
            raise ValueError("dataset.type not specified in config")

        self.logger.info(
            "Loading dataset: %s (split=%s, limit=%s)",
            dataset_type,
            split,
            limit,
        )

        loader = DataloaderCatalog.create(
            dataset_type,
            split=split,
            limit=limit,
            dataset_id=dataset_name,
        )

        documents = loader.load().to_haystack()

        self.logger.info("Loaded %d documents from dataset", len(documents))
        return documents

    def run(self, batch_size: int = 32) -> dict[str, Any]:
        """Execute indexing pipeline.

        Loads documents, generates embeddings, and stores in database.

        Args:
            batch_size: Number of documents to process per batch.

        Returns:
            Dictionary with indexing statistics.
        """
        self.logger.info("Starting indexing pipeline")

        try:
            # Load dataset
            documents = self._load_dataset()

            if not documents:
                self.logger.warning("No documents loaded")
                return {"indexed_count": 0, "status": "empty_dataset"}

            # Generate embeddings in batches
            self.logger.info("Generating embeddings for %d documents", len(documents))
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                self.logger.debug(
                    "Processing batch %d-%d", i, min(i + batch_size, len(documents))
                )

                # Embed batch
                for doc in batch:
                    embedding_result = self.dense_embedder.run(text=doc.content)
                    doc.embedding = embedding_result["embedding"]

                # Store batch
                self._store_documents(batch)

            self.logger.info("Successfully indexed %d documents", len(documents))
            return {
                "indexed_count": len(documents),
                "status": "success",
                "batch_size": batch_size,
            }

        except Exception as e:
            self.logger.error("Error during indexing: %s", str(e))
            return {"indexed_count": 0, "status": "error", "error": str(e)}
