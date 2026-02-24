"""Weaviate MMR indexing pipeline.

This pipeline loads documents from a configured dataset, generates embeddings using
a specified embedding model, creates a Weaviate collection, and indexes the documents.

Weaviate-Specific Considerations:
    - Weaviate combines vector search with semantic (GraphQL) capabilities
    - Schema-based with automatic vectorization support
    - Supports modular AI integrations (vectorization, generative, qna)
    - Can be self-hosted or used as a managed service (WCS)

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create collection: Initialize Weaviate class/collection with schema
    4. Upsert documents: Store vectors and metadata in Weaviate

Configuration (YAML):
    - weaviate.url: Weaviate server URL (e.g., "http://localhost:8080")
    - weaviate.collection_name: Name of the collection/class to create
    - weaviate.recreate: Whether to drop and recreate existing collection
    - embeddings.model: HuggingFace model path for embeddings
    - dataloader.type: Dataset type (e.g., "triviaqa")

Usage:
    >>> pipeline = WeaviateMmrIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class WeaviateMmrIndexingPipeline:
    """Weaviate indexing pipeline for MMR search.

    Loads documents, generates embeddings, creates collection, and indexes.

    This pipeline follows the standard 3-stage indexing pattern:
    1. Load documents from the configured dataset
    2. Generate embeddings using the configured embedder
    3. Create collection and upsert documents to Weaviate

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack document embedder component.
        db: WeaviateVectorDB instance for database operations.
        collection_name: Name of the Weaviate collection/class.
        dimension: Vector dimension (must match embedding model).

    Note:
        Weaviate uses a schema-based approach. The collection is created
        with properties for content and metadata, plus a vector index.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            collection_name=weaviate_config.get("collection_name"),
        )

        self.collection_name = weaviate_config.get("collection_name")
        self.dimension = weaviate_config.get("dimension", 384)

        logger.info("Initialized Weaviate MMR indexing pipeline")

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

        # Step 3: Create or recreate Weaviate collection
        # Collection includes schema with content and metadata properties
        recreate = self.config.get("weaviate", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            recreate=recreate,
        )

        # Step 4: Upsert documents to Weaviate
        num_indexed = self.db.upsert(documents=embedded_docs)
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
