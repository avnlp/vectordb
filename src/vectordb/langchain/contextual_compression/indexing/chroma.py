"""Chroma contextual compression indexing pipeline (LangChain).

This module provides an indexing pipeline for Chroma vector database with
contextual compression support. The pipeline handles document loading,
embedding generation, and Chroma collection management.

Chroma-Specific Considerations:
    Chroma is an embedded vector database that stores data locally. Key
    characteristics relevant to this pipeline:

    - Persistence: Data is stored in a local directory (persist_dir)
    - Collections: Documents are organized into named collections
    - No namespaces: Unlike Pinecone, Chroma doesn't support namespaces
    - Recreation: Collections can be dropped and recreated for fresh indexing

Pipeline Flow:
    1. Load configuration from dict or YAML file
    2. Initialize embedder based on configuration
    3. Connect to Chroma (creates local storage if doesn't exist)
    4. Load documents using configured dataloader
    5. Generate embeddings for all documents
    6. Create/recreate Chroma collection
    7. Upsert documents with embeddings into collection

Configuration Schema:
    Required keys in configuration:
    - chroma.persist_dir: Directory for Chroma persistence
    - chroma.collection_name: Name of the collection to create/use
    - chroma.recreate: Whether to drop and recreate collection (default: false)
    - embedding: Embedding model configuration
    - dataloader: Data source configuration
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


# Module-level logger for indexing operations
logger = logging.getLogger(__name__)


class ChromaContextualCompressionIndexingPipeline:
    """Indexing pipeline for Chroma with contextual compression support.

    This pipeline prepares document collections for contextual compression-based
    retrieval. It handles the complete indexing workflow from document loading
    to Chroma collection population.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for document vectorization
        db: ChromaVectorDB instance for collection management
        collection_name: Name of the Chroma collection to use

    Design Decisions:
        - Configuration-driven: All parameters loaded from config for reproducibility
        - Lazy loading: Documents are loaded during run(), not initialization
        - Batch processing: Embeddings generated for all documents at once
        - Idempotent: Can be run multiple times with recreate=True for fresh starts
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the indexing pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. The configuration must contain 'chroma',
                'embedding', and 'dataloader' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Example:
            >>> pipeline = ChromaContextualCompressionIndexingPipeline("config.yaml")
            >>> # Or with dict
            >>> config = {"chroma": {"persist_dir": "./chroma", ...}, ...}
            >>> pipeline = ChromaContextualCompressionIndexingPipeline(config)
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]

        self.db = ChromaVectorDB(
            persist_dir=chroma_config.get("persist_dir"),
        )

        self.collection_name = chroma_config.get("collection_name")

        logger.info(
            "Initialized Chroma contextual compression indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the indexing pipeline.

        This method performs the complete indexing workflow:
        1. Load documents from configured datasource
        2. Generate embeddings for all documents
        3. Create or recreate the Chroma collection
        4. Upsert documents with embeddings into the collection

        Returns:
            Dictionary containing:
                - 'documents_indexed': Number of documents successfully indexed

        Note:
            If no documents are loaded (empty datasource), returns 0 count
            without raising an error. This allows graceful handling of
            edge cases in batch processing scenarios.
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

        recreate = self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=recreate,
        )

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
