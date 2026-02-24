"""Milvus JSON indexing pipeline for LangChain.

This module provides the indexing pipeline for JSON document storage using Milvus
as the vector store backend. Milvus offers cloud-native JSON indexing with dynamic
field support, enabling flexible schema evolution and structured metadata storage.

The indexing pipeline follows the standard three-phase pattern:
1. Document loading via DataloaderCatalog (supports multiple dataset types)
2. Embedding generation using configurable embedders
3. Vector store indexing with collection management

JSON Indexing Pattern:
    - Text content extracted from JSON for embedding
    - Full JSON structure preserved in document metadata
    - Dynamic field support for evolving JSON schemas
    - Enables both semantic search and structured filtering

Milvus-specific Implementation:
    - Cloud-native architecture with distributed storage
    - Dynamic field support for flexible JSON schemas
    - Collection-based document organization with partitions
    - Batch insert operations for efficient indexing
    - Support for both local and remote Milvus instances

Use Cases:
    - Large-scale product catalogs with evolving attributes
    - Multi-tenant document stores with custom schemas
    - Real-time analytics with JSON metadata filtering
    - Enterprise knowledge bases with complex structures
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class MilvusJsonIndexingPipeline:
    """Milvus indexing pipeline for JSON documents (LangChain).

    Loads JSON documents, generates embeddings for text content, creates
    collection, and indexes documents with preserved JSON structure in metadata.

    Leverages Milvus's dynamic field support to store JSON metadata alongside
    dense vectors, enabling efficient filtering during search operations.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: MilvusVectorDB instance for vector storage operations.
        collection_name: Target Milvus collection for document storage.

    Example:
        >>> config = {
        ...     "dataloader": {"type": "triviaqa", "split": "test"},
        ...     "embeddings": {"model": "all-MiniLM-L6-v2"},
        ...     "milvus": {
        ...         "uri": "http://localhost:19530",
        ...         "collection_name": "json_docs",
        ...     },
        ... }
        >>> pipeline = MilvusJsonIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the embedding model and
        Milvus vector store connection. Supports both local Milvus Lite
        instances and remote Milvus servers.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'milvus' section with uri and collection_name,
                plus embedding configuration.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config path doesn't exist.

        Configuration Schema:
            milvus:
              uri: "http://localhost:19530"
              token: ""  # Optional, for cloud instances
              collection_name: "json_indexed"

            embeddings:
              model: "sentence-transformers/all-MiniLM-L6-v2"
              device: "cpu"

            dataloader:
              type: "triviaqa"
              split: "test"
              limit: 100
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            token=milvus_config.get("token", ""),
        )

        self.collection_name = milvus_config.get("collection_name")

        logger.info("Initialized Milvus JSON indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute JSON document indexing pipeline.

        Performs the complete indexing workflow: loads JSON documents from
        the configured dataset, generates embeddings for text content,
        creates the Milvus collection with appropriate dimension, and
        inserts all documents with their vectors and preserved JSON metadata.

        The pipeline handles empty document sets gracefully and provides
        detailed logging for monitoring indexing progress. Document limits
        from configuration are applied during loading to support testing
        with subsets of large datasets.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed
                - collection_name: Name of the Milvus collection used

        Raises:
            Exception: If embedding generation or vector store operations fail.

        Milvus-specific Notes:
            - Collection created with dynamic field support for JSON metadata
            - Dense vectors stored in primary vector field
            - JSON metadata stored in dynamic fields for flexible querying
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
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            use_sparse=False,
        )
        logger.info("Created Milvus collection: %s", self.collection_name)

        num_indexed = self.db.insert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d JSON documents to Milvus", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "collection_name": self.collection_name,
        }
