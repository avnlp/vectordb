"""Qdrant reranking indexing pipeline.

This module provides an indexing pipeline for preparing document collections
in Qdrant for two-stage retrieval with reranking.

Indexing for Reranking:
    Documents are embedded using bi-encoder models and stored in Qdrant.
    These embeddings enable fast HNSW-based approximate nearest neighbor
    search during the retrieval phase. Cross-encoder reranking operates
    on text content, so only bi-encoder embeddings are stored.

Pipeline Steps:
    1. Load documents from configured data sources
    2. Generate dense embeddings using bi-encoder models
    3. Create or recreate Qdrant collection with vector index
    4. Upsert embedded documents with payload metadata

Qdrant Features:
    - High-performance HNSW index for vector search
    - Rich payload filtering with complex conditions
    - Distributed deployment for horizontal scaling
    - Sparse vector support for hybrid retrieval
    - On-disk and in-memory storage options

Collection Configuration:
    The collection stores vectors with associated payload (metadata).
    HNSW index parameters affect search speed vs recall trade-offs.
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class QdrantRerankingIndexingPipeline:
    """Qdrant indexing pipeline for reranking document collections.

    Prepares document collections for two-stage retrieval by generating
    bi-encoder embeddings and storing them in Qdrant. The indexed vectors
    enable fast approximate nearest neighbor search, while payload metadata
    supports filtering during retrieval.

    Attributes:
        config: Pipeline configuration dict.
        embedder: Bi-encoder component for document embedding generation.
        dimension: Embedding dimension from embedder configuration.
        db: QdrantVectorDB instance for collection management.
        collection_name: Name of the Qdrant collection to create/use.

    Example:
        >>> pipeline = QdrantRerankingIndexingPipeline("qdrant_config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - qdrant: url, collection_name, api_key (for cloud)
                - embedder: Provider, model, dimensions for bi-encoder
                - dataloader: Dataset source and optional limit

        Raises:
            ValueError: If required config sections are missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)
        self.dimension = EmbedderFactory.get_embedding_dimension(self.embedder)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            collection_name=qdrant_config.get("collection_name", "reranking"),
        )

        self.collection_name = qdrant_config.get("collection_name", "reranking")

        logger.info("Initialized Qdrant reranking indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Loads documents from data sources, generates bi-encoder embeddings,
        creates the Qdrant collection with HNSW index, and upserts all
        documents with vectors and payload metadata.

        Returns:
            Dict with 'documents_indexed' count.

        Raises:
            RuntimeError: If embedding generation or Qdrant upsert fails.
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

        recreate = self.config.get("qdrant", {}).get("recreate", False)
        self.db.create_collection(
            dimension=self.dimension,
            recreate=recreate,
        )

        self.db.upsert(documents=embedded_docs)
        num_indexed = len(embedded_docs)
        logger.info("Indexed %d documents to Qdrant", num_indexed)

        return {"documents_indexed": num_indexed}
