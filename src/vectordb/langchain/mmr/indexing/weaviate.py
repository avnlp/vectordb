"""Weaviate MMR indexing pipeline (LangChain).

This module provides a LangChain-native indexing pipeline for preparing
Weaviate vector store collections for MMR-based retrieval operations.

Pipeline Architecture:
    1. Configuration Loading: Validates Weaviate-specific settings
    2. Embedding Generation: Creates dense vector representations
    3. Schema Setup: Initializes Weaviate schema with class definition
    4. Document Upsert: Stores documents with embeddings in Weaviate

MMR Preparation:
    Indexing stores document embeddings which the MMR algorithm uses during
    search to balance query relevance against result diversity. Weaviate's
    modular architecture supports both vector and BM25 hybrid search.

Example:
    >>> from vectordb.langchain.mmr.indexing import WeaviateMMRIndexingPipeline
    >>> pipeline = WeaviateMMRIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    - vectordb.langchain.mmr.search.weaviate: MMR search for Weaviate
    - vectordb.WeaviateVectorDB: Core Weaviate vector database wrapper
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


class WeaviateMMRIndexingPipeline:
    """LangChain indexing pipeline for Weaviate MMR support.

    Prepares Weaviate collections for MMR-based retrieval by indexing
    documents with their dense embeddings. Weaviate's GraphQL interface
    and hybrid search capabilities complement MMR diversity filtering.

    Attributes:
        config: Validated configuration dictionary containing Weaviate
            connection parameters, collection settings, and data source.
        embedder: LangChain embedder instance for vector generation.
        db: WeaviateVectorDB instance for vector store operations.
        collection_name: Name of the Weaviate collection for documents.

    Example:
        >>> config = {
        ...     "weaviate": {
        ...         "url": "http://localhost:8080",
        ...         "collection_name": "MMR",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"dataset": "triviaqa"},
        ... }
        >>> pipeline = WeaviateMMRIndexingPipeline(config)
        >>> pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate MMR indexing pipeline.

        Loads configuration, validates Weaviate-specific settings,
        initializes the embedder, and establishes connection to Weaviate.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'weaviate' section with connection details.

        Raises:
            ValueError: If required Weaviate configuration (url) is missing
                or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Weaviate server.

        Configuration Schema:
            weaviate:
                url: Weaviate server URL (required)
                api_key: Optional API key for authentication
                collection_name: Name for the collection (default: "MMR")
            embedder: Embedder configuration dict
            dataloader: Data source configuration dict
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name", "MMR")

        logger.info("Initialized Weaviate MMR indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts them to Weaviate. Returns statistics about the operation.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Upsert documents to Weaviate collection

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or Weaviate upsert fails.
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
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
