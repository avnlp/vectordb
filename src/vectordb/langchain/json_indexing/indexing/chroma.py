"""Chroma JSON indexing pipeline for LangChain.

This module provides the indexing pipeline for JSON document storage using Chroma
as the vector store backend. JSON indexing enables structured data storage with
vector embeddings, allowing semantic search over JSON documents while preserving
structured fields.

The indexing pipeline follows the standard three-phase pattern:
1. Document loading via DataloaderCatalog (supports multiple dataset types)
2. Embedding generation using configurable embedders
3. Vector store indexing with collection management

JSON Indexing Pattern:
    - Text content extracted from JSON for embedding
    - Full JSON structure preserved in document metadata
    - Nested JSON fields flattened for metadata filtering
    - Enables both semantic search and structured filtering

Chroma-specific Implementation:
    - Local persistent storage with path configuration
    - Collection-based document organization
    - Metadata-rich document storage for JSON field filtering
    - Batch upsert operations for efficient indexing

Use Cases:
    - Product catalogs with attributes (price, category, brand)
    - API documentation with endpoints (method, path, parameters)
    - Configuration files with settings (section, key, value)
    - Structured knowledge bases (entity, relation, attributes)
"""

import logging
import uuid
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class ChromaJsonIndexingPipeline:
    """Chroma indexing pipeline for JSON documents (LangChain).

    Loads JSON documents, generates embeddings for text content, creates
    collection, and indexes documents with preserved JSON structure in metadata.

    The pipeline maintains compatibility with the search pipeline by ensuring
    JSON metadata is properly structured for filtering operations during search.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: ChromaVectorDB instance for vector storage operations.
        collection_name: Target Chroma collection for document storage.

    Example:
        >>> config = {
        ...     "dataloader": {"type": "triviaqa", "split": "test"},
        ...     "embeddings": {"model": "all-MiniLM-L6-v2"},
        ...     "chroma": {"path": "./data", "collection_name": "json_docs"},
        ... }
        >>> pipeline = ChromaJsonIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the embedding model and
        Chroma vector store connection. Supports both local persistent
        Chroma instances and in-memory configurations.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'chroma' section with path and collection_name,
                plus embedding configuration.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config path doesn't exist.

        Configuration Schema:
            chroma:
              path: "./chroma_data"
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
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path"),
            collection_name=chroma_config.get("collection_name"),
        )

        self.collection_name = chroma_config.get("collection_name")

        logger.info("Initialized Chroma JSON indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute JSON document indexing pipeline.

        Performs the complete indexing workflow: loads JSON documents from
        the configured dataset, generates embeddings for text content,
        creates or recreates the Chroma collection, and upserts all
        documents with their vectors and preserved JSON metadata.

        The pipeline handles empty document sets gracefully and provides
        detailed logging for monitoring indexing progress. Document limits
        from configuration are applied during loading to support testing
        with subsets of large datasets.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed
                - collection_name: Name of the Chroma collection used

        Raises:
            Exception: If embedding generation or vector store operations fail.

        JSON Handling:
            - Documents loaded with JSON structure preserved in metadata
            - Text content extracted from JSON for embedding
            - Full JSON structure maintained in document.metadata for filtering
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

        self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            name=self.collection_name,
            get_or_create=True,
        )

        # Build data dictionary with ids, documents, embeddings, and metadatas
        # for upsert
        ids = [str(uuid.uuid4()) for _ in docs]
        upsert_data = {
            "ids": ids,
            "documents": [d.page_content for d in docs],
            "embeddings": embeddings,
            "metadatas": [d.metadata for d in docs],
        }
        self.db.upsert(data=upsert_data)
        num_indexed = len(docs)
        logger.info("Indexed %d JSON documents to Chroma", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "collection_name": self.collection_name,
        }
