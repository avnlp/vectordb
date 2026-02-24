"""Milvus MMR indexing pipeline (LangChain).

This module provides a LangChain-native indexing pipeline for preparing
Milvus vector store collections for MMR-based retrieval operations.

Pipeline Architecture:
    1. Configuration Loading: Validates Milvus-specific settings
    2. Embedding Generation: Creates dense vector representations
    3. Collection Setup: Initializes Milvus collection with schema
    4. Document Upsert: Stores documents with embeddings in Milvus

MMR Preparation:
    While indexing itself doesn't implement MMR, it prepares the collection
    by storing document embeddings. The MMR algorithm operates on these
    embeddings during search time to balance relevance with diversity.

Example:
    >>> from vectordb.langchain.mmr.indexing import MilvusMMRIndexingPipeline
    >>> pipeline = MilvusMMRIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    - vectordb.langchain.mmr.search.milvus: MMR search for Milvus
    - vectordb.MilvusVectorDB: Core Milvus vector database wrapper
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


class MilvusMMRIndexingPipeline:
    """LangChain indexing pipeline for Milvus MMR support.

    Prepares Milvus collections for MMR-based retrieval by indexing
    documents with their embeddings. The embeddings enable both
    standard similarity search and MMR diversity-aware reranking.

    Attributes:
        config: Validated configuration dictionary containing Milvus
            connection parameters, collection settings, and data source.
        embedder: LangChain embedder instance for vector generation.
        db: MilvusVectorDB instance for vector store operations.
        collection_name: Name of the Milvus collection for documents.

    Example:
        >>> config = {
        ...     "milvus": {
        ...         "host": "localhost",
        ...         "port": 19530,
        ...         "collection_name": "mmr_docs",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"dataset": "triviaqa"},
        ... }
        >>> pipeline = MilvusMMRIndexingPipeline(config)
        >>> pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus MMR indexing pipeline.

        Loads configuration, validates Milvus-specific settings,
        initializes the embedder, and establishes connection to Milvus.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'milvus' section with connection details.

        Raises:
            ValueError: If required Milvus configuration (host, collection_name)
                is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Milvus server.

        Configuration Schema:
            milvus:
                host: Milvus server hostname (default: "localhost")
                port: Milvus server port (default: 19530)
                collection_name: Name for the collection (default: "mmr")
            embedder: Embedder configuration dict
            dataloader: Data source configuration dict
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get("collection_name", "mmr")

        logger.info("Initialized Milvus MMR indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts them to Milvus. Returns statistics about the operation.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Upsert documents to Milvus collection

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or Milvus upsert fails.
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
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
