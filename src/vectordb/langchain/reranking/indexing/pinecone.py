"""Pinecone reranking indexing pipeline (LangChain).

This module provides the indexing pipeline for Pinecone vector database
with reranking support. It handles document loading, embedding generation,
and indexing for later retrieval with cross-encoder reranking.

The pipeline is designed for cloud-scale deployments with Pinecone's
managed vector database service, supporting features like:
- Serverless or pod-based index creation
- Namespace-based data isolation
- Metadata filtering support
- Index recreation for fresh indexing runs
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class PineconeReankingIndexingPipeline:
    """Indexing pipeline for Pinecone with reranking support.

    This pipeline loads documents, generates embeddings using the configured
    embedding model, and indexes them into Pinecone for later reranked search.

    The pipeline handles:
    - Configuration loading and validation
    - Embedding model initialization
    - Pinecone connection and index management
    - Document batching and upsert operations

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model instance
        db: PineconeVectorDB instance for database operations
        index_name: Name of the Pinecone index
        namespace: Namespace for document isolation
        dimension: Vector dimension (must match embedding model)

    Example:
        >>> pipeline = PineconeReankingIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")

    Configuration Requirements:
        The config file must specify:
        - pinecone.api_key: Pinecone API key
        - pinecone.index_name: Target index name
        - pinecone.dimension: Vector dimension (e.g., 384, 768, 1536)
        - pinecone.metric: Distance metric (cosine, euclidean, dotproduct)
        - embedder: Embedding model configuration
        - dataloader: Data source configuration
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone indexing pipeline.

        Loads configuration, initializes the embedding model, and establishes
        connection to Pinecone vector database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file.

        Raises:
            ValueError: If required configuration keys are missing or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Pinecone.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone reranking indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        creates the Pinecone index if needed, and upserts all documents.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.
            ValueError: If no documents are found in the data source.

        Pipeline Steps:
            1. Load documents from configured data source
            2. Generate embeddings for all documents
            3. Create or verify Pinecone index exists
            4. Upsert documents with embeddings to Pinecone
            5. Return count of indexed documents
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

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
