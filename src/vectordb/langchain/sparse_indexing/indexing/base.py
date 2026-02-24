"""Base sparse indexing pipeline for LangChain vector databases.

This module provides an abstract base class that encapsulates common logic
for sparse indexing pipelines, including document loading and embedding
generation. Concrete implementations only need to implement database-specific
indexing logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import ConfigLoader, SparseEmbedder


logger = logging.getLogger(__name__)


class BaseSparseIndexingPipeline(ABC):
    """Abstract base class for sparse indexing pipelines.

    This class abstracts the common logic across all sparse indexing pipelines:
    - Loading configuration
    - Loading documents from dataloaders
    - Generating sparse embeddings

    Subclasses must implement:
    - _initialize_db(): Database-specific initialization
    - _index_documents(): Database-specific indexing logic

    Example:
        >>> class MilvusSparseIndexingPipeline(BaseSparseIndexingPipeline):
        ...     def _initialize_db(self) -> None:
        ...         milvus_config = self.config["milvus"]
        ...         self.db = MilvusVectorDB(
        ...             host=milvus_config.get("host", "localhost"),
        ...             port=milvus_config.get("port", 19530),
        ...         )
        ...
        ...     def _index_documents(
        ...         self,
        ...         documents: list[Document],
        ...         sparse_embeddings: list[dict[str, float]],
        ...     ) -> int:
        ...         # Attach embeddings and insert
        ...         for doc, sparse_emb in zip(documents, sparse_embeddings):
        ...             doc.sparse_embedding = sparse_emb
        ...         self.db.insert_documents(documents=documents)
        ...         return len(documents)
    """

    def __init__(
        self,
        config_or_path: dict[str, Any] | str,
        db_config_key: str,
        model_name: str = "naver/splade-v2",
        device: str = "cpu",
    ) -> None:
        """Initialize base sparse indexing pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
            db_config_key: Key in config for database-specific settings
                (e.g., "milvus", "qdrant", "pinecone").
            model_name: Sparse embedding model name (default: naver/splade-v2).
            device: Device for sparse embedding model ("cpu" or "cuda").

        Raises:
            ValueError: If configuration validation fails.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, db_config_key)

        self.db_config_key = db_config_key
        self.embedder = SparseEmbedder(model_name=model_name, device=device)

        # Initialize database-specific components
        self._initialize_db()

        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    def _initialize_db(self) -> None:
        """Initialize database-specific components.

        This method should:
        1. Extract database configuration from self.config[self.db_config_key]
        2. Initialize the database client (self.db)
        3. Set any database-specific attributes (collection_name, etc.)

        Example:
            >>> def _initialize_db(self) -> None:
            ...     milvus_config = self.config["milvus"]
            ...     self.db = MilvusVectorDB(
            ...         host=milvus_config.get("host", "localhost"),
            ...         port=milvus_config.get("port", 19530),
            ...     )
            ...     self.collection_name = milvus_config.get("collection_name")
        """
        pass

    @abstractmethod
    def _index_documents(
        self,
        documents: list[Document],
        sparse_embeddings: list[dict[str, float]],
    ) -> int:
        """Index documents with sparse embeddings to the database.

        This method should:
        1. Prepare documents in database-specific format
        2. Attach sparse embeddings to documents
        3. Insert/upsert documents to the database
        4. Return the number of successfully indexed documents

        Args:
            documents: List of LangChain documents to index.
            sparse_embeddings: List of sparse embeddings (one per document).
                Each sparse embedding is a dict {token_id: weight}.

        Returns:
            Number of documents successfully indexed.

        Example:
            >>> def _index_documents(self, documents, sparse_embeddings):
            ...     for doc, sparse_emb in zip(documents, sparse_embeddings):
            ...         doc.sparse_embedding = sparse_emb
            ...     self.db.insert_documents(documents=documents)
            ...     return len(documents)
        """
        pass

    def _load_documents(self) -> list[Document]:
        """Load documents from the configured dataloader.

        This method handles:
        - Extracting dataloader configuration
        - Creating the appropriate dataloader
        - Loading the dataset
        - Converting to LangChain documents

        Returns:
            List of loaded documents.
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_langchain()
        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _generate_embeddings(self, documents: list[Document]) -> list[dict[str, float]]:
        """Generate sparse embeddings for documents.

        Args:
            documents: List of documents to embed.

        Returns:
            List of sparse embeddings (one per document).
        """
        if not documents:
            logger.warning("No documents to embed")
            return []

        texts = [doc.page_content for doc in documents]
        sparse_embeddings = self.embedder.embed_documents(texts)
        logger.info(f"Generated sparse embeddings for {len(documents)} documents")
        return sparse_embeddings

    def run(self) -> dict[str, Any]:
        """Execute the sparse indexing pipeline.

        This method orchestrates the full indexing workflow:
        1. Load documents from dataloader
        2. Generate sparse embeddings
        3. Index documents to database

        Returns:
            Dict with indexing results, including:
            - documents_indexed: Number of documents indexed
            - status: Pipeline execution status

        Raises:
            Exception: If any step in the pipeline fails.
        """
        # Step 1: Load documents
        documents = self._load_documents()

        if not documents:
            logger.warning("No documents to index")
            return {
                "documents_indexed": 0,
                "status": "completed",
                "reason": "No documents to index",
            }

        # Step 2: Generate sparse embeddings
        sparse_embeddings = self._generate_embeddings(documents)

        # Step 3: Index documents (database-specific)
        num_indexed = self._index_documents(documents, sparse_embeddings)

        logger.info(f"Successfully indexed {num_indexed} documents")
        return {
            "documents_indexed": num_indexed,
            "status": "completed",
        }
