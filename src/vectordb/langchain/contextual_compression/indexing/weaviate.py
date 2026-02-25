"""Weaviate contextual compression indexing pipeline (LangChain).

This module provides an indexing pipeline for Weaviate vector database with
contextual compression support. Weaviate is an open-source vector database
with a schema-based approach and GraphQL interface.

Weaviate-Specific Considerations:
    Weaviate offers unique features for vector storage:

    - Schema-based: Requires class definitions with property types
    - GraphQL interface: Query using GraphQL syntax
    - Modular AI: Supports vectorization modules (text2vec, etc.)
    - Hybrid search: Combines vector and BM25 keyword search
    - Multi-modal: Supports images, audio, video in addition to text

Pipeline Flow:
    1. Load configuration from dict or YAML file
    2. Initialize embedder based on configuration
    3. Connect to Weaviate server (URL + API key)
    4. Load documents using configured dataloader
    5. Generate embeddings for all documents
    6. Create/recreate Weaviate collection (class)
    7. Upsert documents with embeddings into collection

Configuration Schema:
    Required keys:
    - weaviate.url: Weaviate server URL
    - weaviate.api_key: Authentication key (optional for local)
    - weaviate.collection_name: Class/collection to create/use
    - weaviate.recreate: Whether to recreate (default: false)
    - embedding: Embedding model configuration
    - dataloader: Data source configuration
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


# Module-level logger for indexing operations
logger = logging.getLogger(__name__)


class WeaviateContextualCompressionIndexingPipeline:
    """Indexing pipeline for Weaviate with contextual compression support.

    This pipeline prepares document collections for contextual compression-based
    retrieval in Weaviate. It handles the complete indexing workflow from document
    loading to Weaviate class population.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for document vectorization
        db: WeaviateVectorDB instance for class management
        collection_name: Name of the Weaviate class to use

    Design Decisions:
        - URL-based connection: Connects via HTTP URL
        - Schema-aware: Works with Weaviate's class-based schema
        - Flexible authentication: Supports API key or local instances
        - Collection pattern: Uses 'collection' terminology consistently
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the indexing pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'weaviate', 'embedding', and
                'dataloader' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
            ConnectionError: If unable to connect to Weaviate server.

        Example:
            >>> pipeline = WeaviateContextualCompressionIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]

        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name")

        logger.info(
            "Initialized Weaviate contextual compression indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the indexing pipeline.

        This method performs the complete indexing workflow:
        1. Load documents from configured datasource
        2. Generate embeddings for all documents
        3. Create or recreate the Weaviate class
        4. Upsert documents with embeddings into the class

        Returns:
            Dictionary containing:
                - 'documents_indexed': Number of documents successfully indexed

        Note:
            Weaviate class recreation deletes all objects of that class.
            Consider using batch imports for incremental updates in production.
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        dl_type = dl_config.get("type")
        if not dl_type:
            raise ValueError("dataloader.type must be specified in the configuration.")
        loader = DataloaderCatalog.create(
            dl_type,
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

        recreate = self.config.get("weaviate", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=recreate,
        )

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
