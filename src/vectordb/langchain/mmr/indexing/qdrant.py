"""Qdrant MMR indexing pipeline (LangChain).

This module provides a LangChain-native indexing pipeline for preparing
Qdrant vector store collections for MMR-based retrieval operations.

Pipeline Architecture:
    1. Configuration Loading: Validates Qdrant-specific settings
    2. Embedding Generation: Creates dense vector representations
    3. Collection Setup: Initializes Qdrant collection with HNSW indexing
    4. Document Upsert: Stores documents with embeddings in Qdrant

MMR Preparation:
    Indexing stores document embeddings which are later used by the MMR
    algorithm during search. MMR balances query relevance against diversity
    to reduce result redundancy.

Example:
    >>> from vectordb.langchain.mmr.indexing import QdrantMMRIndexingPipeline
    >>> pipeline = QdrantMMRIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    - vectordb.langchain.mmr.search.qdrant: MMR search for Qdrant
    - vectordb.QdrantVectorDB: Core Qdrant vector database wrapper
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class QdrantMMRIndexingPipeline:
    """LangChain indexing pipeline for Qdrant MMR support.

    Prepares Qdrant collections for MMR-based retrieval by indexing
    documents with their dense embeddings. These embeddings enable
    the MMR algorithm to perform diversity-aware reranking at search time.

    Attributes:
        config: Validated configuration dictionary containing Qdrant
            connection parameters, collection settings, and data source.
        embedder: LangChain embedder instance for vector generation.
        db: QdrantVectorDB instance for vector store operations.
        collection_name: Name of the Qdrant collection for documents.

    Example:
        >>> config = {
        ...     "qdrant": {
        ...         "url": "http://localhost:6333",
        ...         "collection_name": "mmr_docs",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"dataset": "triviaqa"},
        ... }
        >>> pipeline = QdrantMMRIndexingPipeline(config)
        >>> pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Qdrant MMR indexing pipeline.

        Loads configuration, validates Qdrant-specific settings,
        initializes the embedder, and establishes connection to Qdrant.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'qdrant' section with connection details.

        Raises:
            ValueError: If required Qdrant configuration (url, collection_name)
                is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.

        Configuration Schema:
            qdrant:
                url: Qdrant server URL (default: "http://localhost:6333")
                api_key: Optional API key for authentication
                collection_name: Name for the collection (default: "mmr")
            embedder: Embedder configuration dict
            dataloader: Data source configuration dict
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name", "mmr")

        logger.info("Initialized Qdrant MMR indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts them to Qdrant. Returns statistics about the operation.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Upsert documents to Qdrant collection

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or Qdrant upsert fails.
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
        logger.info("Indexed %d documents to Qdrant", num_indexed)

        return {"documents_indexed": num_indexed}
