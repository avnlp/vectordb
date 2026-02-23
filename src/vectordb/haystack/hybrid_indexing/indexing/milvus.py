"""Milvus hybrid indexing pipeline.

This module provides hybrid indexing capabilities for Milvus vector database,
enabling efficient storage and retrieval of both dense (semantic) and sparse
(lexical) document embeddings.

Milvus Hybrid Search:
    Milvus provides native hybrid search support with sophisticated sparse
    vector handling and automatic result fusion using RRF (Reciprocal Rank
    Fusion) or other ranking strategies.

    Key Features:
    - Native sparse vector collection fields
    - Built-in RRF (Reciprocal Rank Fusion) ranking
    - Efficient sparse vector storage with inverted index
    - Support for multiple vector fields per collection

Sparse Vector Support:
    Milvus stores sparse vectors as (index, value) pairs in a specialized
    sparse vector field. This enables:
    - Efficient storage of high-dimensional sparse data
    - Fast sparse vector similarity search
    - Seamless hybrid dense+sparse retrieval

    The sparse_embedder generates these (index, weight) representations
    using models like SPLADE that learn term importance.

RRF Fusion Strategy:
    Reciprocal Rank Fusion combines rankings from dense and sparse searches:
    score = sum(1.0 / (k + rank)) where k=60 (default)

    This approach is robust, requires no score normalization, and works
    well even when dense and sparse scores are on different scales.

Example:
    >>> from vectordb.haystack.hybrid_indexing.indexing.milvus import (
    ...     MilvusHybridIndexingPipeline,
    ... )
    >>> indexer = MilvusHybridIndexingPipeline(
    ...     config_path="configs/milvus/triviaqa.yaml"
    ... )
    >>> result = indexer.run()
    >>> print(f"Indexed {result['documents_indexed']} documents to Milvus")
"""

import logging
from typing import Any

from haystack import Document

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class MilvusHybridIndexingPipeline:
    """Milvus hybrid (dense + sparse) indexing pipeline.

    Indexes documents with both dense and sparse embeddings into Milvus,
    enabling hybrid search with RRF ranking. Milvus natively supports
    sparse vectors as first-class citizens alongside dense vectors.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense semantic embeddings.
        sparse_embedder: Optional component for sparse lexical embeddings
            using SPLADE or similar learned sparse models.
        db: MilvusVectorDB instance for document storage.
        collection_name: Name of the Milvus collection.
        dimension: Vector dimension for dense embeddings.
        batch_size: Number of documents per upsert batch.
        recreate: Whether to drop and recreate existing collection.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'milvus' section with connection/collection settings
                and 'embeddings' section. 'sparse' section enables hybrid search.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> pipeline = MilvusHybridIndexingPipeline("configs/milvus.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.dense_embedder = EmbedderFactory.create_document_embedder(self.config)
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_document_embedder(
                self.config
            )

        milvus_config = self.config["milvus"]
        uri = milvus_config.get("uri", "http://localhost:19530")
        token = milvus_config.get("token", "")

        self.db = MilvusVectorDB(uri=uri, token=token)

        self.collection_name = milvus_config.get("collection_name")
        self.dimension = milvus_config.get("dimension", 768)
        self.batch_size = milvus_config.get("batch_size", 100)
        self.recreate = milvus_config.get("recreate", False)

        logger.info("Initialized Milvus hybrid indexing pipeline at %s", uri)

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents, generates both dense and sparse embeddings, creates
        Milvus collection with appropriate schema (dense + optional sparse
        fields), and upserts documents in batches.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Total number of documents indexed
            - db: Database identifier ("milvus")
            - collection_name: Name of the target Milvus collection

        Raises:
            RuntimeError: If collection creation or upsert operations fail.
        """
        logger.info("Starting Milvus hybrid indexing pipeline")

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
            return {"documents_indexed": 0, "db": "milvus"}

        embedded_docs = self._embed_documents(documents)
        logger.info("Embedded %d documents", len(embedded_docs))

        use_sparse = bool(self.sparse_embedder)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            use_sparse=use_sparse,
            recreate=self.recreate,
        )
        logger.info(
            "Created collection %s (sparse=%s)", self.collection_name, use_sparse
        )

        for i in range(0, len(embedded_docs), self.batch_size):
            batch = embedded_docs[i : i + self.batch_size]
            self.db.upsert(
                documents=batch,
                collection_name=self.collection_name,
            )
            logger.debug(
                "Indexed batch %d/%d (%d docs)",
                i // self.batch_size + 1,
                (len(embedded_docs) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )

        logger.info(
            "Indexed %d documents to Milvus collection %s",
            len(embedded_docs),
            self.collection_name,
        )

        return {
            "documents_indexed": len(embedded_docs),
            "db": "milvus",
            "collection_name": self.collection_name,
        }

    def _embed_documents(self, documents: list[Document]) -> list[Document]:
        """Generate dense and sparse embeddings for documents.

        Dense embeddings capture semantic meaning using neural networks.
        Sparse embeddings (if configured) capture lexical term importance
        using learned sparse models like SPLADE, producing sparse vectors
        as (token_index, weight) pairs.

        Args:
            documents: List of Haystack Document objects to embed.

        Returns:
            Documents with dense embeddings attached and optional sparse
            embeddings stored in the embedding metadata.
        """
        dense_result = self.dense_embedder.run(documents=documents)
        embedded_docs = dense_result["documents"]
        logger.debug("Generated dense embeddings for %d documents", len(embedded_docs))

        if self.sparse_embedder:
            sparse_result = self.sparse_embedder.run(documents=embedded_docs)
            embedded_docs = sparse_result["documents"]
            logger.debug(
                "Generated sparse embeddings for %d documents", len(embedded_docs)
            )

        return embedded_docs
