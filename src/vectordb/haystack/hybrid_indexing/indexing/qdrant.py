"""Qdrant hybrid indexing pipeline.

This module provides hybrid indexing capabilities for Qdrant vector database,
enabling storage and retrieval of both dense (semantic) and sparse (lexical)
document embeddings with native Qdrant hybrid search support.

Qdrant Hybrid Search:
    Qdrant provides comprehensive hybrid search capabilities through its
    native hybrid query API, supporting both dense and sparse vectors in
    a unified interface.

    Key Features:
    - Native sparse vector storage with efficient indexing
    - Built-in hybrid search query type
    - Automatic fusion of dense and sparse results
    - Support for multiple vectors per point with named vectors

Sparse Vector Storage:
    Qdrant stores sparse vectors as dictionaries mapping dimension indices
    to non-zero values. This representation is space-efficient for high-
    dimensional sparse data like SPLADE embeddings.

    The sparse_embedder component generates these representations using
    learned sparse models that predict term importance across a vocabulary.

Hybrid Query Execution:
    Qdrant's hybrid search performs both dense and sparse lookups,
    then fuses results internally using sophisticated ranking algorithms.
    This provides a seamless hybrid search experience without manual
    result fusion.

Example:
    >>> from vectordb.haystack.hybrid_indexing.indexing.qdrant import (
    ...     QdrantHybridIndexingPipeline,
    ... )
    >>> indexer = QdrantHybridIndexingPipeline(
    ...     config_path="configs/qdrant/triviaqa.yaml"
    ... )
    >>> result = indexer.run()
    >>> print(f"Indexed {result['documents_indexed']} documents to Qdrant")
"""

import logging
from typing import Any

from haystack import Document

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class QdrantHybridIndexingPipeline:
    """Qdrant hybrid (dense + sparse) indexing pipeline.

    Indexes documents with both dense and sparse embeddings into Qdrant,
    enabling hybrid search with native Qdrant hybrid retriever capabilities.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense semantic embeddings.
        sparse_embedder: Optional component for sparse lexical embeddings.
        db: QdrantVectorDB instance for document storage.
        collection_name: Name of the Qdrant collection.
        dimension: Vector dimension for dense embeddings.
        batch_size: Number of documents per upsert batch.
        recreate: Whether to drop and recreate existing collection.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'qdrant' section with connection/collection settings.
                'sparse' section enables hybrid indexing with learned sparse models.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> pipeline = QdrantHybridIndexingPipeline("configs/qdrant.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        # Initialize dense embedder for semantic vector search
        self.dense_embedder = EmbedderFactory.create_document_embedder(self.config)

        # Initialize sparse embedder if configured for lexical search
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_document_embedder(
                self.config
            )

        # Configure Qdrant connection
        qdrant_config = self.config["qdrant"]
        url = qdrant_config.get("url", "http://localhost:6333")
        api_key = qdrant_config.get("api_key")
        path = qdrant_config.get("path")
        self.collection_name = qdrant_config.get("collection_name")

        config_dict = {
            "qdrant": {
                "url": url,
                "api_key": api_key,
                "path": path,
                "collection_name": self.collection_name,
            }
        }
        self.db = QdrantVectorDB(config=config_dict)

        self.dimension = qdrant_config.get("dimension", 768)
        self.batch_size = qdrant_config.get("batch_size", 100)
        self.recreate = qdrant_config.get("recreate", False)

        logger.info("Initialized Qdrant hybrid indexing pipeline at %s", url)

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents, generates both dense and sparse embeddings,
        creates Qdrant collection with sparse support if needed, and
        upserts documents in batches.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Total number of documents indexed
            - db: Database identifier ("qdrant")
            - collection_name: Name of the target Qdrant collection

        Raises:
            RuntimeError: If collection creation or upsert operations fail.
        """
        logger.info("Starting Qdrant hybrid indexing pipeline")

        # Load documents from configured datasource
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
            return {"documents_indexed": 0, "db": "qdrant"}

        # Generate dense and optional sparse embeddings
        embedded_docs = self._embed_documents(documents)
        logger.info("Embedded %d documents", len(embedded_docs))

        # Create collection with sparse vector support if configured
        use_sparse = bool(self.sparse_embedder)
        self.db.create_collection(
            dimension=self.dimension,
            use_sparse=use_sparse,
            recreate=self.recreate,
        )
        logger.info(
            "Created Qdrant collection %s (sparse=%s)", self.collection_name, use_sparse
        )

        # Upsert documents in batches
        for i in range(0, len(embedded_docs), self.batch_size):
            batch = embedded_docs[i : i + self.batch_size]
            self.db.index_documents(documents=batch)
            logger.debug(
                "Indexed batch %d/%d (%d docs)",
                i // self.batch_size + 1,
                (len(embedded_docs) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )

        logger.info(
            "Indexed %d documents to Qdrant collection %s",
            len(embedded_docs),
            self.collection_name,
        )

        return {
            "documents_indexed": len(embedded_docs),
            "db": "qdrant",
            "collection_name": self.collection_name,
        }

    def _embed_documents(self, documents: list[Document]) -> list[Document]:
        """Generate dense and sparse embeddings for documents.

        Processes documents through dense embedder (required) and optional
        sparse embedder. Sparse embeddings use learned models like SPLADE
        to produce sparse vectors representing term importance.

        Args:
            documents: List of Haystack Document objects to embed.

        Returns:
            Documents with dense embeddings and optional sparse embeddings
            suitable for Qdrant's hybrid search.
        """
        # Generate dense semantic embeddings (required)
        dense_result = self.dense_embedder.run(documents=documents)
        embedded_docs = dense_result["documents"]
        logger.debug("Generated dense embeddings for %d documents", len(embedded_docs))

        # Generate sparse lexical embeddings if configured
        if self.sparse_embedder:
            sparse_result = self.sparse_embedder.run(documents=embedded_docs)
            embedded_docs = sparse_result["documents"]
            logger.debug(
                "Generated sparse embeddings for %d documents", len(embedded_docs)
            )

        return embedded_docs
