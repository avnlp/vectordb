"""Chroma MMR indexing pipeline (LangChain).

This module provides a LangChain-native indexing pipeline for preparing
Chroma vector store collections for MMR-based retrieval operations.

Pipeline Architecture:
    1. Configuration Loading: Validates Chroma-specific settings
    2. Embedding Generation: Creates dense vector representations
    3. Collection Setup: Initializes Chroma collection in local storage
    4. Document Upsert: Stores documents with embeddings in Chroma

MMR Preparation:
    Indexing stores document embeddings which the MMR algorithm uses during
    search to balance query relevance against result diversity. Chroma's
    local storage makes this ideal for development and testing.

Example:
    >>> from vectordb.langchain.mmr.indexing import ChromaMMRIndexingPipeline
    >>> pipeline = ChromaMMRIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    - vectordb.langchain.mmr.search.chroma: MMR search for Chroma
    - vectordb.ChromaVectorDB: Core Chroma vector database wrapper
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


class ChromaMMRIndexingPipeline:
    """LangChain indexing pipeline for Chroma MMR support.

    Prepares Chroma collections for MMR-based retrieval by indexing
    documents with their dense embeddings. Chroma's embedded nature
    makes it ideal for local development and testing scenarios.

    Attributes:
        config: Validated configuration dictionary containing Chroma
            storage path, collection settings, and data source.
        embedder: LangChain embedder instance for vector generation.
        db: ChromaVectorDB instance for vector store operations.
        collection_name: Name of the Chroma collection for documents.

    Example:
        >>> config = {
        ...     "chroma": {
        ...         "path": "./chroma_data",
        ...         "collection_name": "mmr_docs",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"dataset": "triviaqa"},
        ... }
        >>> pipeline = ChromaMMRIndexingPipeline(config)
        >>> pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma MMR indexing pipeline.

        Loads configuration, validates Chroma-specific settings,
        initializes the embedder, and prepares local Chroma storage.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'chroma' section with storage path.

        Raises:
            ValueError: If required Chroma configuration (path, collection_name)
                is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            PermissionError: If unable to write to the Chroma storage path.

        Configuration Schema:
            chroma:
                path: Directory path for Chroma storage (default: "./chroma_data")
                collection_name: Name for the collection (default: "mmr")
            embedder: Embedder configuration dict
            dataloader: Data source configuration dict
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path", "./chroma_data"),
        )

        self.collection_name = chroma_config.get("collection_name", "mmr")

        logger.info("Initialized Chroma MMR indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts them to Chroma. Returns statistics about the operation.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Upsert documents to Chroma collection

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or Chroma upsert fails.
            ValueError: If no documents are found in the data source.

        Example:
            >>> result = pipeline.run()
            >>> assert result["documents_indexed"] > 0
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
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
