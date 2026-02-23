"""Weaviate agentic RAG indexing pipeline (LangChain).

This module provides the indexing pipeline for agentic RAG using Weaviate as the
vector store backend. Weaviate's modular AI-native architecture provides native
vector search combined with GraphQL interfaces, enabling sophisticated agentic
RAG workflows that blend semantic and structured queries.

The indexing pipeline leverages Weaviate's schema-driven approach:
1. Document loading via DataloaderCatalog with schema mapping
2. Embedding generation with vectorizer configuration
3. Collection creation with property definitions for hybrid search

Weaviate-Specific Advantages for Agentic RAG:
    - Native multi-modal support enables agent processing of images, audio, video
    - Modular AI integrations (OpenAI, Cohere, HuggingFace) simplify embedding setup
    - GraphQL interface allows complex queries combining vector and BM25 search
    - Reference properties enable document relationship traversal during agent reasoning

Architecture Notes:
    Weaviate's schema-first approach requires upfront property definition but
    enables powerful hybrid search capabilities. The agentic search pipeline can
    leverage Weaviate's hybrid search to combine dense vector similarity with
    sparse BM25 scoring, improving retrieval for keyword-heavy agent queries.
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


class WeaviateAgenticRAGIndexingPipeline:
    """Weaviate indexing pipeline for agentic RAG (LangChain).

    Loads documents, generates embeddings, creates Weaviate collection, and indexes
    documents for use in agentic RAG pipelines. Weaviate's AI-native design provides
    integrated embedding and search capabilities that simplify agentic RAG deployment.

    The pipeline supports both Weaviate Cloud (WCS) and self-hosted deployments
    through URL and API key configuration.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: WeaviateVectorDB instance for vector storage operations.
        collection_name: Target Weaviate collection (class) for document storage.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the Weaviate connection.
        Supports both Weaviate Cloud Service and self-hosted instances.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'weaviate' section with url, optional api_key,
                and collection_name (class name) settings.

        Raises:
            ValueError: If required Weaviate configuration is missing.

        Example:
            >>> pipeline = WeaviateAgenticRAGIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents to Weaviate")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Configure Weaviate connection
        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config.get("url"),
            api_key=weaviate_config.get("api_key"),
        )

        # In Weaviate, collections are called "classes"
        self.collection_name = weaviate_config.get("collection_name")

        logger.info("Initialized Weaviate agentic RAG indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs complete indexing workflow: loads documents, generates embeddings,
        creates or recreates the Weaviate collection (class) with proper schema,
        and upserts all documents.

        Weaviate's schema defines properties that can be vectorized and/or
        used for filtering. The agentic search pipeline can leverage these
        properties for hybrid search combining vector similarity with keyword
        matching through Weaviate's BM25 implementation.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed

        Raises:
            Exception: If collection creation or document upsert fails.
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

        # Create collection (class) with schema definition
        # Schema properties enable hybrid search and metadata filtering
        recreate = self.config.get("weaviate", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=recreate,
        )

        # Upsert documents to Weaviate class
        # Documents are stored as objects with vector and property representations
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
