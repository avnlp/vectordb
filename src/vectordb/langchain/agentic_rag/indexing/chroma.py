"""Chroma agentic RAG indexing pipeline (LangChain).

This module provides the indexing pipeline for agentic RAG using Chroma as the
vector store backend. Agentic RAG requires the same document infrastructure as
standard RAG, but the retrieval and generation phases are orchestrated by an
agent that can make dynamic decisions about search, reflection, and answer
generation.

The indexing pipeline follows the standard three-phase pattern:
1. Document loading via DataloaderCatalog (supports multiple dataset types)
2. Embedding generation using configurable embedders
3. Vector store indexing with collection management

Architecture Notes:
    The indexing pipeline is decoupled from the search pipeline to allow
    independent scaling and deployment. Documents are indexed once and can be
    queried multiple times by the agentic search pipeline. Chroma's local
    embedded mode makes it ideal for development and testing of agentic RAG
    workflows before deploying to production vector stores.

    The pipeline uses ChromaVectorDB for persistence, with support for:
    - Collection creation with configurable recreation policies
    - Batch document upserts with embeddings
    - Metadata-rich document storage for filtering during agentic search
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class ChromaAgenticRAGIndexingPipeline:
    """Chroma indexing pipeline for agentic RAG (LangChain).

    Loads documents, generates embeddings, creates collection, and indexes
    documents for use in agentic RAG pipelines. This pipeline prepares the
    document store that the agentic search pipeline will query during its
    iterative search-reflect-generate cycles.

    The agentic RAG pattern requires high-quality document embeddings since
    the agent may perform multiple retrieval operations during a single query
    session. Poor embeddings lead to suboptimal retrieval, causing the agent
    to either generate from insufficient context or loop excessively.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: ChromaVectorDB instance for vector storage operations.
        collection_name: Target Chroma collection for document storage.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the embedding model and
        Chroma vector store connection. The pipeline supports both local
        persistent Chroma instances and in-memory configurations.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'chroma' section with path and collection_name,
                plus embedding configuration.

        Raises:
            ValueError: If required configuration is missing or invalid.

        Example:
            >>> pipeline = ChromaAgenticRAGIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        # Initialize embedder based on configuration
        # The embedder choice significantly impacts retrieval quality for agentic RAG
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Configure Chroma connection with local persistence
        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path"),
            collection_name=chroma_config.get("collection_name"),
        )

        self.collection_name = chroma_config.get("collection_name")

        logger.info("Initialized Chroma agentic RAG indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs the complete indexing workflow: loads documents from the
        configured dataset, generates embeddings, creates or recreates the
        Chroma collection, and upserts all documents with their vectors.

        The pipeline handles empty document sets gracefully and provides
        detailed logging for monitoring indexing progress. Document limits
        from configuration are applied during loading to support testing
        with subsets of large datasets.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed

        Raises:
            Exception: If embedding generation or vector store operations fail.
        """
        # Load documents with optional limit for testing/development
        # The limit parameter enables rapid iteration with smaller datasets
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

        # Generate embeddings for all loaded documents
        # EmbedderHelper handles batching and error recovery
        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d documents", len(docs))

        # Create or recreate collection based on configuration
        # Recreate=True is useful for development but should be False in production
        recreate = self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=recreate,
        )

        # Upsert documents with embeddings to Chroma
        # Upsert operation handles both new inserts and updates to existing docs
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
