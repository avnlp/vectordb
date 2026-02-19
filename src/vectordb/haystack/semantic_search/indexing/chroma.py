"""Chroma semantic search indexing pipeline.

This pipeline provides document indexing for Chroma's embedded vector database,
enabling semantic similarity search through dense vector embeddings for local
development and small-to-medium scale deployments.

Chroma-Specific Considerations:
    - Local embedded vector database with SQLite persistence
    - Simple API with collection-based organization
    - Supports cosine similarity search (default metric)
    - Good for development and small-to-medium datasets
    - No external service dependencies

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create collection: Initialize Chroma collection
    4. Insert documents: Store vectors and metadata in Chroma

Configuration (YAML):
    Required sections:
        - chroma.host: Chroma server host (default: localhost)
        - chroma.port: Chroma server port (default: 8000)
        - chroma.collection_name: Name of the collection to create
        - embeddings.model: HuggingFace model path for embeddings
        - dataloader.type: Dataset type (e.g., "triviaqa")

    Optional settings:
        - chroma.persist_directory: Directory for Chroma persistence
        - chroma.recreate: Whether to drop and recreate existing collection
        - dataloader.limit: Optional limit on documents to process

    Example config:
        chroma:
          host: "localhost"
          port: 8000
          collection_name: "semantic-search"
          persist_directory: "./chroma_data"
          recreate: false
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        dataloader:
          type: "triviaqa"
          limit: 1000

Usage:
    >>> from vectordb.haystack.semantic_search import ChromaSemanticIndexingPipeline
    >>> pipeline = ChromaSemanticIndexingPipeline("config.yaml")
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

    Both implementations use the same underlying ChromaVectorDB class for
database operations, ensuring consistent behavior across frameworks.

Note:
    Chroma stores data locally in persist_directory. For production
    deployments, consider migrating to a server-based database.
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class ChromaSemanticIndexingPipeline:
    """Chroma indexing pipeline for semantic search.

    Loads documents, generates embeddings, creates collection, and indexes.

    This pipeline follows the standard 3-stage indexing pattern:
    1. Load documents from the configured dataset
    2. Generate embeddings using the configured embedder
    3. Create collection and insert documents to Chroma

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack document embedder component.
        db: ChromaVectorDB instance for database operations.
        collection_name: Name of the Chroma collection.

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
            host=chroma_config.get("host", "localhost"),
            port=chroma_config.get("port", 8000),
        )

        self.collection_name = chroma_config["collection_name"]

        logger.info("Initialized Chroma indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute the full indexing pipeline.

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

        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=self.config.get("chroma", {}).get("recreate", False),
        )

        self.db.insert_documents(
            documents=embedded_docs,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", len(embedded_docs))

        return {"documents_indexed": len(embedded_docs)}
