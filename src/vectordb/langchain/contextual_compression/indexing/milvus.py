"""Milvus contextual compression indexing pipeline (LangChain).

This module provides an indexing pipeline for Milvus vector database with
contextual compression support. Milvus is a scalable open-source vector
database designed for high-performance similarity search.

Milvus-Specific Considerations:
    Milvus offers enterprise-grade features for vector storage:

    - Distributed architecture: Supports horizontal scaling
    - Partitions: Logical data organization within collections
    - Multiple index types: IVF, HNSW, etc. for performance tuning
    - Rich metadata: Supports complex filtering expressions
    - Self-hosted or cloud: Can run locally or use managed service

Pipeline Flow:
    1. Load configuration from dict or YAML file
    2. Initialize embedder based on configuration
    3. Connect to Milvus server (host:port)
    4. Load documents using configured dataloader
    5. Generate embeddings for all documents
    6. Create/recreate Milvus collection
    7. Upsert documents with embeddings into collection

Configuration Schema:
    Required keys:
    - milvus.host: Milvus server host
    - milvus.port: Milvus server port
    - milvus.db_name: Database name (optional)
    - milvus.collection_name: Collection to create/use
    - milvus.dimension: Vector dimension (default: 384)
    - milvus.recreate: Whether to drop and recreate (default: false)
    - embedding: Embedding model configuration
    - dataloader: Data source configuration
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


# Module-level logger for indexing operations
logger = logging.getLogger(__name__)


class MilvusContextualCompressionIndexingPipeline:
    """Indexing pipeline for Milvus with contextual compression support.

    This pipeline prepares document collections for contextual compression-based
    retrieval in Milvus. It handles the complete indexing workflow from document
    loading to Milvus collection population.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for document vectorization
        db: MilvusVectorDB instance for collection management
        collection_name: Name of the Milvus collection to use
        dimension: Vector dimension for the collection

    Design Decisions:
        - Server connection: Connects to Milvus server via host/port
        - Collection-based: Uses Milvus collections (similar to tables)
        - Configurable dimension: Matches embedding model output size
        - Idempotent operations: Supports recreation for clean starts
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the indexing pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'milvus', 'embedding', and
                'dataloader' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
            ConnectionError: If unable to connect to Milvus server.

        Example:
            >>> pipeline = MilvusContextualCompressionIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]

        self.db = MilvusVectorDB(
            host=milvus_config.get("host"),
            port=milvus_config.get("port"),
            db_name=milvus_config.get("db_name"),
        )

        self.collection_name = milvus_config.get("collection_name")
        self.dimension = milvus_config.get("dimension", 384)

        logger.info(
            "Initialized Milvus contextual compression indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the indexing pipeline.

        This method performs the complete indexing workflow:
        1. Load documents from configured datasource
        2. Generate embeddings for all documents
        3. Create or recreate the Milvus collection
        4. Upsert documents with embeddings into the collection

        Returns:
            Dictionary containing:
                - 'documents_indexed': Number of documents successfully indexed

        Note:
            Milvus collection recreation drops all existing data. Use with
            caution in production. Consider using partitions for incremental
            updates instead of full recreation.
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

        recreate = self.config.get("milvus", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            recreate=recreate,
        )

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
