"""Chroma MMR indexing pipeline.

This pipeline loads documents from a configured dataset, generates embeddings using
a specified embedding model, creates a Chroma collection, and indexes the documents.

Chroma-Specific Considerations:
    - Local embedded vector database with SQLite persistence
    - Simple API with collection-based organization
    - Supports cosine similarity search (default metric)
    - Good for development and small-to-medium datasets

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create collection: Initialize Chroma collection
    4. Upsert documents: Store vectors and metadata in Chroma

Configuration (YAML):
    - chroma.persist_directory: Directory for Chroma persistence
    - chroma.collection_name: Name of the collection to create
    - chroma.recreate: Whether to drop and recreate existing collection
    - embeddings.model: HuggingFace model path for embeddings
    - dataloader.type: Dataset type (e.g., "triviaqa")
    - dataloader.limit: Optional limit on documents to process

Usage:
    >>> pipeline = ChromaMmrIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class ChromaMmrIndexingPipeline:
    """Chroma indexing pipeline for MMR search.

    Loads documents, generates embeddings, creates collection, and indexes.

    This pipeline follows the standard 3-stage indexing pattern:
    1. Load documents from the configured dataset
    2. Generate embeddings using the configured embedder
    3. Create collection and upsert documents to Chroma

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack document embedder component.
        db: ChromaVectorDB instance for database operations.
        collection_name: Name of the Chroma collection.
        dimension: Vector dimension (must match embedding model).

    Note:
        Chroma stores data locally in persist_directory. For production
        deployments, consider migrating to a server-based database.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_directory=chroma_config.get("persist_directory"),
            collection_name=chroma_config.get("collection_name"),
        )

        self.collection_name = chroma_config.get("collection_name")
        self.dimension = chroma_config.get("dimension", 384)

        logger.info("Initialized Chroma MMR indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Returns:
            Dict with 'documents_indexed' count.
        """
        # Step 1: Load documents from configured dataset
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_haystack()
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        # Step 2: Generate embeddings for all documents
        embedded_docs = self.embedder.run(documents=documents)["documents"]
        logger.info("Generated embeddings for %d documents", len(embedded_docs))

        # Step 3: Create or recreate Chroma collection
        recreate = self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=recreate,
        )

        # Step 4: Upsert documents to Chroma
        num_indexed = self.db.upsert(documents=embedded_docs)
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
