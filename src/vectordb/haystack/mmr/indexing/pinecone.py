"""Pinecone MMR indexing pipeline.

This pipeline loads documents from a configured dataset, generates embeddings using
a specified embedding model, creates a Pinecone index, and indexes the documents.

Pinecone-Specific Considerations:
    - Pinecone is a managed cloud service requiring API key authentication
    - Indexes are created with a specified dimension and metric
      (cosine, dotproduct, euclidean)
    - Supports namespaces for logical partitioning within an index
    - Serverless and pod-based deployment options available

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create index: Initialize Pinecone index with proper dimension/metric
    4. Upsert documents: Store vectors and metadata in Pinecone

Configuration (YAML):
    - pinecone.api_key: Pinecone API key
    - pinecone.index_name: Name of the index to create
    - pinecone.namespace: Optional namespace for document organization
    - pinecone.metric: Similarity metric (cosine, dotproduct, euclidean)
    - pinecone.recreate: Whether to drop and recreate existing index
    - embeddings.model: HuggingFace model path for embeddings
    - dataloader.type: Dataset type (e.g., "triviaqa")

Usage:
    >>> pipeline = PineconeMmrIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class PineconeMmrIndexingPipeline:
    """Pinecone indexing pipeline for MMR search.

    Loads documents, generates embeddings, creates index, and indexes.

    This pipeline follows the standard 3-stage indexing pattern:
    1. Load documents from the configured dataset
    2. Generate embeddings using the configured embedder
    3. Create index and upsert documents to Pinecone

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack document embedder component.
        db: PineconeVectorDB instance for database operations.
        index_name: Name of the Pinecone index.
        namespace: Optional namespace for document organization.
        dimension: Vector dimension (must match embedding model).

    Note:
        Pinecone indexes cannot be modified after creation (dimension/metric
        are immutable). The recreate flag drops and recreates the index.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone MMR indexing pipeline")

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

        # Step 3: Create or recreate Pinecone index
        # Index dimension and metric are immutable after creation
        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        # Step 4: Upsert documents to Pinecone
        # Namespace provides logical partitioning within the index
        num_indexed = self.db.upsert(
            documents=embedded_docs,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
