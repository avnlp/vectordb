"""Pinecone JSON indexing pipeline for LangChain.

This module provides the indexing pipeline for JSON document storage using Pinecone
as the vector store backend. Pinecone offers managed vector search with native
JSON metadata support, enabling scalable semantic search with structured filtering.

The indexing pipeline follows the standard three-phase pattern:
1. Document loading via DataloaderCatalog (supports multiple dataset types)
2. Embedding generation using configurable embedders
3. Vector store indexing with upsert operations

JSON Indexing Pattern:
    - Text content extracted from JSON for embedding
    - Full JSON structure preserved in document metadata
    - Nested JSON fields flattened for metadata filtering
    - Enables both semantic search and structured filtering

Pinecone-specific Implementation:
    - Fully managed cloud service with automatic scaling
    - Native metadata filtering on JSON fields
    - Namespace-based document organization
    - Serverless and pod-based deployment options
    - Automatic index management and dimension detection

Use Cases:
    - Production-scale semantic search applications
    - Multi-tenant SaaS with isolated namespaces
    - Real-time recommendation systems
    - Enterprise search with access control filtering
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


class PineconeJsonIndexingPipeline:
    """Pinecone indexing pipeline for JSON documents (LangChain).

    Loads JSON documents, generates embeddings for text content, and
    upserts documents with preserved JSON structure in metadata.

    Leverages Pinecone's managed infrastructure for automatic scaling
    and index management, with namespace support for multi-tenant scenarios.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: PineconeVectorDB instance for vector storage operations.
        namespace: Target Pinecone namespace for document storage.

    Example:
        >>> config = {
        ...     "dataloader": {"type": "triviaqa", "split": "test"},
        ...     "embeddings": {"model": "all-MiniLM-L6-v2"},
        ...     "pinecone": {
        ...         "api_key": "your-api-key",
        ...         "index_name": "json_docs",
        ...         "namespace": "default",
        ...     },
        ... }
        >>> pipeline = PineconeJsonIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the embedding model and
        Pinecone vector store connection. Requires Pinecone API key
        and index configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'pinecone' section with api_key and index_name,
                plus embedding configuration.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config path doesn't exist.

        Configuration Schema:
            pinecone:
              api_key: "your-api-key"  # Or use PINECONE_API_KEY env var
              index_name: "json_indexed"
              namespace: "default"  # Optional
              environment: "us-west-2"  # Optional, for pod-based indexes

            embeddings:
              model: "sentence-transformers/all-MiniLM-L6-v2"
              device: "cpu"

            dataloader:
              type: "triviaqa"
              split: "test"
              limit: 100
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        self.db = PineconeVectorDB(config=self.config)

        pinecone_config = self.config.get("pinecone", {})
        self.namespace = pinecone_config.get("namespace")

        logger.info("Initialized Pinecone JSON indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute JSON document indexing pipeline.

        Performs the complete indexing workflow: loads JSON documents from
        the configured dataset, generates embeddings for text content, and
        upserts all documents with their vectors and preserved JSON metadata.

        Pinecone automatically creates the index if it doesn't exist, using
        the dimension from the generated embeddings. The pipeline handles
        empty document sets gracefully and provides detailed logging.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed
                - index_name: Name of the Pinecone index used
                - namespace: Namespace where documents were indexed

        Raises:
            Exception: If embedding generation or upsert operations fail.

        Pinecone-specific Notes:
            - Index created automatically with correct dimension
            - Metadata stored as JSON for filtering during search
            - Namespace enables logical separation of document sets
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
            return {
                "documents_indexed": 0,
                "index_name": self.config.get("pinecone", {}).get(
                    "index_name", "default"
                ),
                "namespace": self.namespace,
            }

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d JSON documents", len(docs))

        num_indexed = self.db.upsert_documents(
            documents=docs,
            embeddings=embeddings,
            namespace=self.namespace,
        )
        logger.info("Indexed %d JSON documents to Pinecone", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "index_name": self.config.get("pinecone", {}).get("index_name", "default"),
            "namespace": self.namespace,
        }
