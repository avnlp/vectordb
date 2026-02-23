"""Qdrant reranking indexing pipeline (LangChain).

This module provides the indexing pipeline for Qdrant vector database
with reranking support. Qdrant is a high-performance vector similarity
search engine with a focus on efficiency and scalability.

The pipeline supports:
- Local or remote Qdrant instances
- Collection-based organization with configurable parameters
- Efficient HNSW indexing for fast approximate search
- Rich payload (metadata) support for filtering
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class QdrantReankingIndexingPipeline:
    """Indexing pipeline for Qdrant with reranking support.

    This pipeline loads documents, generates embeddings, and indexes them
    into Qdrant for later reranked search.

    Qdrant is ideal for:
    - High-performance vector search applications
    - Hybrid deployments (local development to production clusters)
    - Applications requiring rich filtering capabilities
    - Cost-effective self-hosted solutions

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model instance
        db: QdrantVectorDB instance for database operations
        collection_name: Name of the Qdrant collection

    Example:
        >>> pipeline = QdrantReankingIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents to Qdrant")

    Configuration Requirements:
        The config file must specify:
        - qdrant.url: Qdrant server URL (default: http://localhost:6333)
        - qdrant.api_key: API key for authenticated instances (optional)
        - qdrant.collection_name: Target collection name (default: "reranking")
        - embedder: Embedding model configuration
        - dataloader: Data source configuration
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Qdrant indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the Qdrant vector database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Qdrant.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name", "reranking")

        logger.info("Initialized Qdrant reranking indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts all documents into the Qdrant collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.

        Pipeline Steps:
            1. Load documents from configured data source
            2. Generate embeddings for all documents using embedder
            3. Upsert documents with embeddings to Qdrant collection
            4. Return count of indexed documents
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
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d documents", len(docs))

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Qdrant", num_indexed)

        return {"documents_indexed": num_indexed}
