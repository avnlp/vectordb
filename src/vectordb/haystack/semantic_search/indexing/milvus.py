"""Milvus semantic search indexing pipeline.

This pipeline provides document indexing for Milvus/Zilliz vector database,
enabling semantic similarity search through dense vector embeddings with
high performance and scalability.

Milvus-Specific Considerations:
    - Milvus supports both local (Lite) and server deployments
    - Collections must specify dimension matching the embedding model
    - Supports dynamic schema for flexible metadata
    - Uses HNSW index for efficient approximate nearest neighbor search
    - Partition keys enable efficient multi-tenancy

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create collection: Initialize Milvus collection with proper schema
    4. Insert documents: Store vectors and metadata in Milvus

Configuration (YAML):
    Required sections:
        - milvus.uri: Milvus server URI (e.g., "http://localhost:19530")
        - milvus.collection_name: Name of the collection to create
        - embeddings.model: HuggingFace model path for embeddings
        - dataloader.type: Dataset type (e.g., "triviaqa")

    Optional settings:
        - milvus.token: API token for Zilliz Cloud authentication
        - milvus.dimension: Vector dimension (default: 384)
        - milvus.metric: Similarity metric (cosine, euclidean, inner_product)
        - milvus.recreate: Whether to drop and recreate existing collection
        - dataloader.limit: Optional limit on documents to process

    Example config:
        milvus:
          uri: "http://localhost:19530"
          token: ""  # Empty for local Milvus
          collection_name: "semantic-search"
          dimension: 384
          metric: "cosine"
          recreate: false
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        dataloader:
          type: "triviaqa"
          limit: 1000

Usage:
    >>> from vectordb.haystack.semantic_search import MilvusSemanticIndexingPipeline
    >>> pipeline = MilvusSemanticIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")

Comparison with LangChain:
    Haystack Integration (this module):
        - Uses native Haystack Document format and embedders
        - Pipeline-based architecture with clear data flow
        - Built-in dataset loading through DataloaderCatalog

    LangChain Integration (vectordb.langchain):
        - Uses LangChain Document format
        - Chain-based composition
        - More flexible but requires more configuration

    Both implementations use the same underlying MilvusVectorDB class for
database operations, ensuring consistent behavior across frameworks.

Note:
    The recreate flag in config controls whether to drop and recreate
the collection. Use with caution in production.
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class MilvusSemanticIndexingPipeline:
    """Milvus indexing pipeline for semantic search.

        Loads documents, generates embeddings, creates collection, and indexes.

        This pipeline follows the standard 3-stage indexing pattern:
        1. Load documents from the configured dataset
        2. Generate embeddings using the configured embedder
        3. Create collection and insert documents to Milvus

    Attributes:
            config: Validated configuration dictionary.
            embedder: Haystack document embedder component.
            db: MilvusVectorDB instance for database operations.
            collection_name: Name of the Milvus collection.
            dimension: Vector dimension (must match embedding model).
            metric: Similarity metric for vector comparison.

    Note:
            The recreate flag in config controls whether to drop and recreate
    the collection. Use with caution in production.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            token=milvus_config.get("token", ""),
        )

        self.collection_name = milvus_config["collection_name"]
        self.dimension = milvus_config.get("dimension", 384)
        self.metric = milvus_config.get("metric", "cosine")

        logger.info("Initialized Milvus indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Returns:
            Dict with 'documents_indexed' count.
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_haystack()
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        embedded_docs = self.embedder.run(documents=documents)["documents"]
        logger.info("Generated embeddings for %d documents", len(embedded_docs))

        recreate = self.config.get("milvus", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            metric_type=self.metric,
            recreate=recreate,
        )

        # Insert documents
        self.db.insert_documents(
            documents=embedded_docs,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Milvus", len(embedded_docs))

        return {"documents_indexed": len(embedded_docs)}
