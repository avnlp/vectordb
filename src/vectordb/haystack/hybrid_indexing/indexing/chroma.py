"""Chroma hybrid indexing pipeline.

This module provides hybrid indexing capabilities for Chroma vector database,
enabling storage of both dense (semantic) and sparse (lexical) embeddings.

Chroma Hybrid Search:
    Chroma supports hybrid search by storing both dense and sparse vectors
    in the same collection. The sparse vector support allows for learned
    sparse representations like SPLADE alongside traditional dense embeddings.

    Key Features:
    - Native sparse vector storage in Chroma collections
    - Separate search paths for dense and sparse queries
    - Manual fusion required (RRF or linear combination)
    - Persistent or in-memory storage options

Sparse Embedding Storage:
    When sparse embeddings are enabled, Chroma creates dual storage:
    1. Dense vectors in the main embedding space
    2. Sparse vectors in a separate sparse index

    This enables independent dense and sparse searches that can be
    fused at query time for hybrid results.

Example:
    >>> from vectordb.haystack.hybrid_indexing.indexing.chroma import (
    ...     ChromaHybridIndexingPipeline,
    ... )
    >>> indexer = ChromaHybridIndexingPipeline(
    ...     config_path="configs/chroma/triviaqa.yaml"
    ... )
    >>> result = indexer.run()
    >>> print(f"Indexed {result['documents_indexed']} documents to Chroma")
"""

import logging
from typing import Any

from haystack import Document

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class ChromaHybridIndexingPipeline:
    """Chroma hybrid indexing pipeline.

    Indexes documents with both dense and sparse embeddings into Chroma,
    enabling hybrid search capabilities. Chroma stores sparse vectors
    separately from dense vectors for independent retrieval paths.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense semantic embeddings.
        sparse_embedder: Optional component for sparse lexical embeddings.
        db: ChromaVectorDB instance for document storage.
        collection_name: Name of the Chroma collection.
        dimension: Vector dimension for dense embeddings.
        batch_size: Number of documents per upsert batch.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'chroma' section with connection/collection settings
                and 'embeddings' section for dense embedding configuration.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.dense_embedder = EmbedderFactory.create_document_embedder(self.config)
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_document_embedder(
                self.config
            )

        chroma_config = self.config["chroma"]
        host = chroma_config.get("host", "localhost")
        port = chroma_config.get("port", 8000)
        persistent = chroma_config.get("persistent", False)
        path = chroma_config.get("path")

        config_dict = {
            "chroma": {
                "host": host,
                "port": port,
                "persistent": persistent,
                "path": path,
            }
        }
        self.db = ChromaVectorDB(config=config_dict)

        self.collection_name = chroma_config.get("collection_name")
        self.dimension = chroma_config.get("dimension", 768)
        self.batch_size = chroma_config.get("batch_size", 100)

        logger.info("Initialized Chroma hybrid indexing pipeline at %s:%d", host, port)

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents, generates embeddings, creates Chroma collection with
        optional sparse support, and upserts documents in batches.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Total number of documents indexed
            - db: Database identifier ("chroma")
            - collection_name: Name of the target Chroma collection
        """
        logger.info("Starting Chroma hybrid indexing pipeline")

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
            return {"documents_indexed": 0, "db": "chroma"}

        embedded_docs = self._embed_documents(documents)
        logger.info("Embedded %d documents", len(embedded_docs))

        use_sparse = bool(self.sparse_embedder)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            use_sparse=use_sparse,
        )
        logger.info(
            "Created Chroma collection %s (sparse=%s)", self.collection_name, use_sparse
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
            "Indexed %d documents to Chroma collection %s",
            len(embedded_docs),
            self.collection_name,
        )

        return {
            "documents_indexed": len(embedded_docs),
            "db": "chroma",
            "collection_name": self.collection_name,
        }

    def _embed_documents(self, documents: list[Document]) -> list[Document]:
        """Generate dense and sparse embeddings for documents.

        Processes documents through dense embedder (required) and optional
        sparse embedder. Dense embeddings capture semantic meaning while
        sparse embeddings capture lexical term importance for hybrid search.

        Args:
            documents: List of Haystack Document objects to embed.

        Returns:
            Documents with dense embeddings and optional sparse embeddings.
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
