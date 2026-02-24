"""Qdrant JSON indexing pipeline for LangChain.

This module provides the indexing pipeline for JSON document storage using Qdrant
as the vector store backend. Qdrant offers advanced vector search with native
JSON payload support, enabling sophisticated filtering and semantic search.

The indexing pipeline follows the standard three-phase pattern:
1. Document loading via DataloaderCatalog (supports multiple dataset types)
2. Embedding generation using configurable embedders
3. Vector store indexing with collection management

JSON Indexing Pattern:
    - Text content extracted from JSON for embedding
    - Full JSON structure preserved in document payload
    - Nested JSON fields accessible via payload filtering
    - Enables both semantic search and structured filtering

Qdrant-specific Implementation:
    - Self-hosted or cloud deployment options
    - Native JSON payload storage with filtering support
    - Collection-based document organization
    - Advanced filtering with nested field access
    - HNSW index for efficient approximate nearest neighbor search

Use Cases:
    - On-premise enterprise search deployments
    - Multi-modal search with complex filtering
    - Real-time recommendation with user preferences
    - Knowledge graphs with structured relationships
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


class QdrantJsonIndexingPipeline:
    """Qdrant indexing pipeline for JSON documents (LangChain).

    Loads JSON documents, generates embeddings for text content, creates
    collection, and indexes documents with preserved JSON structure in payload.

    Leverages Qdrant's native payload (metadata) support to store JSON
    structures, enabling powerful filtering capabilities during search
    including nested field access and complex filter conditions.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: QdrantVectorDB instance for vector storage operations.
        collection_name: Target Qdrant collection for document storage.

    Example:
        >>> config = {
        ...     "dataloader": {"type": "triviaqa", "split": "test"},
        ...     "embeddings": {"model": "all-MiniLM-L6-v2"},
        ...     "qdrant": {
        ...         "url": "http://localhost:6333",
        ...         "collection_name": "json_docs",
        ...     },
        ... }
        >>> pipeline = QdrantJsonIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the embedding model and
        Qdrant vector store connection. Supports both local Qdrant
        instances and remote Qdrant Cloud deployments.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'qdrant' section with url and collection_name,
                plus embedding configuration.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config path doesn't exist.

        Configuration Schema:
            qdrant:
              url: "http://localhost:6333"
              api_key: ""  # Optional, for Qdrant Cloud
              collection_name: "json_indexed"
              grpc_port: 6334  # Optional, for gRPC connection

            embeddings:
              model: "sentence-transformers/all-MiniLM-L6-v2"
              device: "cpu"

            dataloader:
              type: "triviaqa"
              split: "test"
              limit: 100
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config.get("qdrant", {})
        self.db = QdrantVectorDB(config=self.config)

        self.collection_name = qdrant_config.get("collection_name")

        logger.info("Initialized Qdrant JSON indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute JSON document indexing pipeline.

        Performs the complete indexing workflow: loads JSON documents from
        the configured dataset, generates embeddings for text content,
        creates the Qdrant collection with appropriate dimension, and
        upserts all documents with their vectors and preserved JSON payload.

        The pipeline handles empty document sets gracefully and provides
        detailed logging for monitoring indexing progress. Document limits
        from configuration are applied during loading to support testing
        with subsets of large datasets.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed
                - collection_name: Name of the Qdrant collection used

        Raises:
            Exception: If embedding generation or vector store operations fail.

        Qdrant-specific Notes:
            - Collection created with vector dimension from embeddings
            - JSON payload stored natively for filtering
            - Payload indexes can be created for frequently filtered fields
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
        logger.info("Loaded %d JSON documents", len(documents))

        if not documents:
            logger.warning("No JSON documents to index")
            return {"documents_indexed": 0, "collection_name": self.collection_name}

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d JSON documents", len(docs))

        dimension = len(embeddings[0]) if embeddings else 384
        self.db.create_collection(dimension=dimension)
        logger.info("Created Qdrant collection: %s", self.collection_name)

        num_indexed = self.db.upsert_documents(
            documents=docs,
            embeddings=embeddings,
        )
        logger.info("Indexed %d JSON documents to Qdrant", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "collection_name": self.collection_name,
        }
