"""Milvus MMR indexing pipeline.

This pipeline loads documents from a configured dataset, generates embeddings using
a specified embedding model, creates a Milvus collection, and indexes the documents.

Milvus-Specific Considerations:
    - Milvus supports both local (Lite) and server deployments
    - Collections must specify dimension matching the embedding model
    - Supports dynamic schema for flexible metadata
    - Uses HNSW index for efficient approximate nearest neighbor search

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create collection: Initialize Milvus collection with proper schema
    4. Upsert documents: Store vectors and metadata in Milvus

Configuration (YAML):
    - milvus.uri: Milvus server URI (e.g., "http://localhost:19530")
    - milvus.collection_name: Name of the collection to create
    - milvus.recreate: Whether to drop and recreate existing collection
    - embeddings.model: HuggingFace model path for embeddings
    - dataloader.type: Dataset type (e.g., "triviaqa")
    - dataloader.limit: Optional limit on documents to process

Usage:
    >>> pipeline = MilvusMmrIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class MilvusMmrIndexingPipeline:
    """Milvus indexing pipeline for MMR search.

    Loads documents, generates embeddings, creates collection, and indexes.

    This pipeline follows the standard 3-stage indexing pattern:
    1. Load documents from the configured dataset
    2. Generate embeddings using the configured embedder
    3. Create collection and upsert documents to Milvus

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack document embedder component.
        db: MilvusVectorDB instance for database operations.
        collection_name: Name of the Milvus collection.
        dimension: Vector dimension (must match embedding model).

    Note:
        The recreate flag in config controls whether to drop and recreate
        the collection. Use with caution in production.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config["uri"],
            collection_name=milvus_config.get("collection_name"),
        )

        self.collection_name = milvus_config.get("collection_name")
        self.dimension = milvus_config.get("dimension", 384)

        logger.info("Initialized Milvus MMR indexing pipeline")

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

        # Step 3: Create or recreate Milvus collection
        # Collection schema includes vector field and dynamic metadata
        recreate = self.config.get("milvus", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            recreate=recreate,
        )

        # Step 4: Upsert documents to Milvus
        num_indexed = self.db.upsert(documents=embedded_docs)
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
