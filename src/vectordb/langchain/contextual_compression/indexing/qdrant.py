"""Qdrant contextual compression indexing pipeline (LangChain).

This module provides an indexing pipeline for Qdrant vector database with
contextual compression support. Qdrant is a high-performance vector database
with advanced filtering capabilities.

Qdrant-Specific Considerations:
    Qdrant offers unique features for vector storage and retrieval:

    - Payload-based: Stores rich JSON payloads with vectors
    - Filtering: Advanced filtering on payload fields during search
    - Collections: Organizes vectors into named collections
    - Distance metrics: Supports cosine, euclidean, dot product
    - Cloud or self-hosted: Flexible deployment options

Pipeline Flow:
    1. Load configuration from dict or YAML file
    2. Initialize embedder based on configuration
    3. Connect to Qdrant server (URL + API key)
    4. Load documents using configured dataloader
    5. Generate embeddings for all documents
    6. Create/recreate Qdrant collection
    7. Upsert documents with embeddings into collection

Configuration Schema:
    Required keys:
    - qdrant.url: Qdrant server URL
    - qdrant.api_key: Authentication key (optional for local)
    - qdrant.collection_name: Collection to create/use
    - qdrant.dimension: Vector dimension (default: 384)
    - qdrant.recreate: Whether to recreate (default: false)
    - embedding: Embedding model configuration
    - dataloader: Data source configuration
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


# Module-level logger for indexing operations
logger = logging.getLogger(__name__)


class QdrantContextualCompressionIndexingPipeline:
    """Indexing pipeline for Qdrant with contextual compression support.

    This pipeline prepares document collections for contextual compression-based
    retrieval in Qdrant. It handles the complete indexing workflow from document
    loading to Qdrant collection population.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for document vectorization
        db: QdrantVectorDB instance for collection management
        collection_name: Name of the Qdrant collection to use
        dimension: Vector dimension for the collection

    Design Decisions:
        - URL-based connection: Connects via HTTP/gRPC URL
        - Payload storage: Stores document metadata as JSON payloads
        - Collection-based: Uses Qdrant collections for organization
        - Flexible authentication: Supports API key or local instances
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the indexing pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'qdrant', 'embedding', and
                'dataloader' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
            ConnectionError: If unable to connect to Qdrant server.

        Example:
            >>> pipeline = QdrantContextualCompressionIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]

        self.db = QdrantVectorDB(
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name")
        self.dimension = qdrant_config.get("dimension", 384)

        logger.info(
            "Initialized Qdrant contextual compression indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the indexing pipeline.

        This method performs the complete indexing workflow:
        1. Load documents from configured datasource
        2. Generate embeddings for all documents
        3. Create or recreate the Qdrant collection
        4. Upsert documents with embeddings into the collection

        Returns:
            Dictionary containing:
                - 'documents_indexed': Number of documents successfully indexed

        Note:
            Qdrant collection recreation is fast but deletes all data.
            Consider using payload-based filtering for incremental updates
            in production scenarios.
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

        recreate = self.config.get("qdrant", {}).get("recreate", False)
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
        logger.info("Indexed %d documents to Qdrant", num_indexed)

        return {"documents_indexed": num_indexed}
