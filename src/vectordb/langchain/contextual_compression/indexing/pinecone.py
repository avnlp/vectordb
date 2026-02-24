"""Pinecone contextual compression indexing pipeline (LangChain).

This module provides an indexing pipeline for Pinecone vector database with
contextual compression support. The pipeline handles document loading,
embedding generation, and Pinecone index management.

Pinecone-Specific Considerations:
    Pinecone is a managed cloud vector database with distinct characteristics:

    - Serverless/Hosted: No local storage, requires API key
    - Indexes: Top-level containers with fixed dimension and metric
    - Namespaces: Logical partitions within an index for multi-tenancy
    - Metadata: Supports rich metadata filtering during queries
    - Recreation: Indexes can be deleted and recreated (operation is slow)

Pipeline Flow:
    1. Load configuration from dict or YAML file
    2. Initialize embedder based on configuration
    3. Connect to Pinecone using API key
    4. Load documents using configured dataloader
    5. Generate embeddings for all documents
    6. Create/recreate Pinecone index (if needed)
    7. Upsert documents with embeddings into namespace

Configuration Schema:
    Required keys:
    - pinecone.api_key: Pinecone API authentication
    - pinecone.index_name: Name of the Pinecone index
    - pinecone.namespace: Namespace for document organization (default: "")
    - pinecone.dimension: Vector dimension (default: 384)
    - pinecone.metric: Distance metric (default: "cosine")
    - pinecone.recreate: Whether to recreate index (default: false)
    - embedding: Embedding model configuration
    - dataloader: Data source configuration
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


# Module-level logger for indexing operations
logger = logging.getLogger(__name__)


class PineconeContextualCompressionIndexingPipeline:
    """Indexing pipeline for Pinecone with contextual compression support.

    This pipeline prepares document collections for contextual compression-based
    retrieval in Pinecone. It handles the complete indexing workflow from
    document loading to Pinecone index population.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for document vectorization
        db: PineconeVectorDB instance for index management
        index_name: Name of the Pinecone index to use
        namespace: Namespace within the index for document organization
        dimension: Vector dimension for the index

    Design Decisions:
        - Namespace support: Uses namespaces for logical document separation
        - Index dimension: Configurable to match embedding model output
        - Metric selection: Cosine similarity by default, configurable for use case
        - Batch upserts: Efficient bulk loading into Pinecone
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the indexing pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'pinecone', 'embedding', and
                'dataloader' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
            KeyError: If pinecone.api_key is not provided in configuration.

        Example:
            >>> pipeline = PineconeContextualCompressionIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents")
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

        logger.info(
            "Initialized Pinecone contextual compression indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the indexing pipeline.

        This method performs the complete indexing workflow:
        1. Load documents from configured datasource
        2. Generate embeddings for all documents
        3. Create or recreate the Pinecone index
        4. Upsert documents with embeddings into the namespace

        Returns:
            Dictionary containing:
                - 'documents_indexed': Number of documents successfully indexed

        Note:
            Index recreation in Pinecone is a slow operation. Use with caution
            in production environments. Consider using namespaces for
            incremental updates instead of full recreation.
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
