"""Pinecone MMR indexing pipeline (LangChain).

This module provides a LangChain-native indexing pipeline for preparing
Pinecone vector indexes for MMR-based retrieval operations.

Pipeline Architecture:
    1. Configuration Loading: Validates Pinecone-specific settings
    2. Embedding Generation: Creates dense vector representations
    3. Index Creation: Initializes Pinecone index with metadata
    4. Document Upsert: Stores documents with embeddings in Pinecone

MMR Preparation:
    Indexing stores document embeddings which the MMR algorithm uses during
    search to balance query relevance against result diversity. Pinecone's
    serverless architecture provides scalable MMR search capabilities.

Example:
    >>> from vectordb.langchain.mmr.indexing import PineconeMMRIndexingPipeline
    >>> pipeline = PineconeMMRIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    - vectordb.langchain.mmr.search.pinecone: MMR search for Pinecone
    - vectordb.PineconeVectorDB: Core Pinecone vector database wrapper
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


class PineconeMMRIndexingPipeline:
    """LangChain indexing pipeline for Pinecone MMR support.

    Prepares Pinecone indexes for MMR-based retrieval by indexing documents
    with their dense embeddings. Pinecone's managed service provides
    production-ready MMR search with automatic scaling.

    Attributes:
        config: Validated configuration dictionary containing Pinecone
            API credentials, index settings, and data source.
        embedder: LangChain embedder instance for vector generation.
        db: PineconeVectorDB instance for vector store operations.
        index_name: Name of the Pinecone index for documents.
        namespace: Pinecone namespace for document organization.
        dimension: Vector dimension matching the embedder output.

    Example:
        >>> config = {
        ...     "pinecone": {
        ...         "api_key": "pc-api-...",
        ...         "index_name": "mmr-docs",
        ...         "dimension": 384,
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"dataset": "triviaqa"},
        ... }
        >>> pipeline = PineconeMMRIndexingPipeline(config)
        >>> pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone MMR indexing pipeline.

        Loads configuration, validates Pinecone-specific settings,
        initializes the embedder, and establishes connection to Pinecone.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'pinecone' section with API key and index details.

        Raises:
            ValueError: If required Pinecone configuration (api_key) is missing
                or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Pinecone API.

        Configuration Schema:
            pinecone:
                api_key: Pinecone API key (required)
                index_name: Name for the index
                namespace: Namespace for documents (default: "")
                dimension: Vector dimension (default: 384)
                metric: Distance metric (default: "cosine")
                recreate: Whether to recreate index if exists (default: False)
            embedder: Embedder configuration dict
            dataloader: Data source configuration dict
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone MMR indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        creates the Pinecone index, and upserts documents. Returns statistics
        about the operation.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Create Pinecone index with appropriate settings
            4. Upsert documents to Pinecone index

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or Pinecone upsert fails.
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

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
