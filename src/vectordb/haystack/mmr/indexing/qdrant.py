"""Qdrant MMR indexing pipeline.

This pipeline loads documents from a configured dataset, generates embeddings using
a specified embedding model, creates a Qdrant collection, and indexes the documents.

Qdrant-Specific Considerations:
    - Qdrant supports both local (in-memory or disk) and server deployments
    - Collections are created with a specified dimension and distance metric
    - Supports rich metadata filtering and payload storage
    - Offers both gRPC and HTTP APIs

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create collection: Initialize Qdrant collection with HNSW index
    4. Upsert documents: Store vectors and metadata in Qdrant

Configuration (YAML):
    - qdrant.url: Qdrant server URL (e.g., "http://localhost:6333")
    - qdrant.api_key: Optional API key for authenticated servers
    - qdrant.collection_name: Name of the collection to create
    - qdrant.recreate: Whether to drop and recreate existing collection
    - embeddings.model: HuggingFace model path for embeddings
    - dataloader.type: Dataset type (e.g., "triviaqa")

Usage:
    >>> pipeline = QdrantMmrIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class QdrantMmrIndexingPipeline:
    """Qdrant indexing pipeline for MMR search.

    Loads documents, generates embeddings, creates collection, and indexes.

    This pipeline follows the standard 3-stage indexing pattern:
    1. Load documents from the configured dataset
    2. Generate embeddings using the configured embedder
    3. Create collection and upsert documents to Qdrant

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack document embedder component.
        db: QdrantVectorDB instance for database operations.
        collection_name: Name of the Qdrant collection.
        dimension: Vector dimension (must match embedding model).

    Note:
        Qdrant supports both local (in-memory/disk) and server modes.
        The url parameter determines the deployment mode.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name"),
        )

        self.collection_name = qdrant_config.get("collection_name")
        self.dimension = qdrant_config.get("dimension", 384)

        logger.info("Initialized Qdrant MMR indexing pipeline")

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

        # Step 3: Create or recreate Qdrant collection
        # Collection uses HNSW index for efficient vector search
        recreate = self.config.get("qdrant", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            recreate=recreate,
        )

        # Step 4: Upsert documents to Qdrant
        # Payload stores document content and metadata for filtering
        num_indexed = self.db.upsert(documents=embedded_docs)
        logger.info("Indexed %d documents to Qdrant", num_indexed)

        return {"documents_indexed": num_indexed}
