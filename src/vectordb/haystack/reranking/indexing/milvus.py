"""Milvus reranking indexing pipeline.

This module provides an indexing pipeline for preparing document collections
in Milvus for two-stage retrieval with reranking.

Indexing for Reranking:
    Documents are embedded using bi-encoder models and stored in Milvus.
    These embeddings enable fast GPU-accelerated or CPU-based approximate
    nearest neighbor search during retrieval. Cross-encoder reranking
    operates on text, so only bi-encoder embeddings need indexing.

Pipeline Steps:
    1. Load documents from configured data sources
    2. Generate dense embeddings using bi-encoder models
    3. Create or recreate Milvus collection with vector index
    4. Upsert embedded documents with scalar fields for filtering

Milvus Capabilities:
    - Billion-scale vector storage with distributed architecture
    - Multiple index types: IVF_FLAT, IVF_PQ, HNSW, DiskANN
    - GPU acceleration for large-scale similarity search
    - Rich attribute filtering and hybrid dense-sparse search
    - Automatic data management with compaction and garbage collection

Index Selection:
    - HNSW: Best for high recall, moderate build time
    - IVF: Good for large datasets, tunable speed-recall trade-off
    - DiskANN: For memory-constrained scenarios
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class MilvusRerankingIndexingPipeline:
    """Milvus indexing pipeline for reranking document collections.

    Prepares document collections for two-stage retrieval by generating
    bi-encoder embeddings and storing them in Milvus. The distributed
    architecture supports large-scale collections with GPU acceleration.

    Attributes:
        config: Pipeline configuration dict.
        embedder: Bi-encoder component for document embedding generation.
        dimension: Embedding dimension from embedder model.
        db: MilvusVectorDB instance for collection management.
        collection_name: Name of the Milvus collection to create/use.

    Example:
        >>> pipeline = MilvusRerankingIndexingPipeline("milvus_config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - milvus: uri, collection_name, token (for cloud)
                - embedder: Provider, model, dimensions for bi-encoder
                - dataloader: Dataset source and optional limit

        Raises:
            ValueError: If required config sections are missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)
        self.dimension = EmbedderFactory.get_embedding_dimension(self.embedder)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            collection_name=milvus_config.get("collection_name", "reranking"),
        )

        self.collection_name = milvus_config.get("collection_name", "reranking")

        logger.info("Initialized Milvus reranking indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Loads documents from data sources, generates bi-encoder embeddings,
        creates the Milvus collection with appropriate index, and upserts
        all documents with vectors and scalar fields.

        Returns:
            Dict with 'documents_indexed' count.

        Raises:
            RuntimeError: If embedding generation or Milvus upsert fails.
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
            dimension=self.dimension,
            recreate=recreate,
        )

        self.db.upsert(documents=embedded_docs)
        num_indexed = len(embedded_docs)
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
