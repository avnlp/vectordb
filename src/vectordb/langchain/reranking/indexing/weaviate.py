"""Weaviate reranking indexing pipeline (LangChain).

This module provides the indexing pipeline for Weaviate vector database
with reranking support. Weaviate is an AI-native vector database with
built-in semantic search capabilities and GraphQL interface.

The pipeline supports:
- Cloud or self-hosted Weaviate instances
- Collection-based document organization with schema flexibility
- Integration with Weaviate's native reranking modules
- Rich metadata and property support
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class WeaviateReankingIndexingPipeline:
    """Indexing pipeline for Weaviate with reranking support.

    This pipeline loads documents, generates embeddings, and indexes them
    into Weaviate for later reranked search.

    Weaviate is ideal for:
    - Applications requiring GraphQL interface
    - Complex data models with rich metadata
    - Hybrid search combining vector and BM25
    - Production deployments with Weaviate Cloud

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model instance
        db: WeaviateVectorDB instance for database operations
        collection_name: Name of the Weaviate collection/class

    Example:
        >>> pipeline = WeaviateReankingIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents to Weaviate")

    Configuration Requirements:
        The config file must specify:
        - weaviate.url: Weaviate instance URL
        - weaviate.api_key: API key (optional for local instances)
        - weaviate.collection_name: Target collection name (default: "Reranking")
        - embedder: Embedding model configuration
        - dataloader: Data source configuration
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the Weaviate vector database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Weaviate.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name", "Reranking")

        logger.info("Initialized Weaviate reranking indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts all documents into the Weaviate collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.

        Pipeline Steps:
            1. Load documents from configured data source
            2. Generate embeddings for all documents using embedder
            3. Upsert documents with embeddings to Weaviate collection
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
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
