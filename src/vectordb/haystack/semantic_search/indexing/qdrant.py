"""Qdrant semantic search indexing pipeline.

This pipeline provides document indexing for Qdrant vector database,
enabling semantic similarity search through dense vector embeddings with
rich metadata filtering and flexible deployment options.

Qdrant-Specific Considerations:
    - Qdrant supports both local (in-memory or disk) and server deployments
    - Collections are created with a specified dimension and distance metric
    - Supports rich metadata filtering and payload storage
    - Offers both gRPC and HTTP APIs
    - Quantization support for memory-efficient storage

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create collection: Initialize Qdrant collection with HNSW index
    4. Insert documents: Store vectors and metadata in Qdrant

Configuration (YAML):
    Required sections:
        - qdrant.url: Qdrant server URL (e.g., "http://localhost:6333")
        - qdrant.collection_name: Name of the collection to create
        - embeddings.model: HuggingFace model path for embeddings
        - dataloader.type: Dataset type (e.g., "triviaqa")

    Optional settings:
        - qdrant.api_key: Optional API key for authenticated servers
        - qdrant.dimension: Vector dimension (default: 384)
        - qdrant.recreate: Whether to drop and recreate existing collection
        - dataloader.limit: Optional limit on documents to process

    Example config:
        qdrant:
          url: "http://localhost:6333"
          api_key: ""  # Empty for local Qdrant
          collection_name: "semantic-search"
          dimension: 384
          recreate: false
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        dataloader:
          type: "triviaqa"
          limit: 1000

Usage:
    >>> from vectordb.haystack.semantic_search import QdrantSemanticIndexingPipeline
    >>> pipeline = QdrantSemanticIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")

Comparison with LangChain:
    Haystack Integration (this module):
        - Uses native Haystack Document format and embedders
        - Pipeline-based architecture with clear data flow
        - Built-in dataset loading through DataloaderCatalog

    LangChain Integration (vectordb.langchain):
        - Uses LangChain Document format
        - Chain-based composition
        - More flexible but requires more configuration

    Both implementations use the same underlying QdrantVectorDB class for
database operations, ensuring consistent behavior across frameworks.

Note:
    Qdrant supports both local (in-memory/disk) and server modes.
The url parameter determines the deployment mode.
"""

import logging
from typing import Any

from vectordb import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class QdrantSemanticIndexingPipeline:
    """Qdrant indexing pipeline for semantic search.

        Loads documents, generates embeddings, creates collection, and indexes.

        This pipeline follows the standard 3-stage indexing pattern:
        1. Load documents from the configured dataset
        2. Generate embeddings using the configured embedder
        3. Create collection and insert documents to Qdrant

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
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key", ""),
        )

        self.collection_name = qdrant_config["collection_name"]
        self.dimension = qdrant_config.get("dimension", 384)

        logger.info("Initialized Qdrant indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Returns:
            Dict with 'documents_indexed' count.
        """
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

        embedded_docs = self.embedder.run(documents=documents)["documents"]
        logger.info("Generated embeddings for %d documents", len(embedded_docs))

        recreate = self.config.get("qdrant", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            recreate=recreate,
        )

        # Insert documents
        self.db.insert_documents(
            documents=embedded_docs,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Qdrant", len(embedded_docs))

        return {"documents_indexed": len(embedded_docs)}
