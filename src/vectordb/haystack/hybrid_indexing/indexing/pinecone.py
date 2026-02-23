"""Pinecone hybrid indexing pipeline.

This module provides hybrid indexing capabilities for Pinecone vector database,
enabling both dense (semantic) and sparse (lexical) document embeddings.

Pinecone Hybrid Search:
    Pinecone natively supports sparse vectors through its sparse_vector field,
    allowing direct storage and querying of sparse embeddings alongside dense
    vectors in the same index.

    Key Features:
    - Native sparse vector storage in metadata
    - Hybrid search with configurable alpha weighting
    - Automatic fusion of dense and sparse results
    - Namespace support for multi-tenant scenarios

Sparse Embedding Handling:
    Sparse embeddings are generated using SPLADE or similar learned sparse
    models. These embeddings represent documents as sparse vectors where
    non-zero values indicate term importance. The sparse_embedder component
    processes documents in batches, attaching sparse representations to
    the embedding field for storage.

Example:
    >>> from vectordb.haystack.hybrid_indexing.indexing.pinecone import (
    ...     PineconeHybridIndexingPipeline,
    ... )
    >>> indexer = PineconeHybridIndexingPipeline(
    ...     config_path="configs/pinecone/triviaqa.yaml"
    ... )
    >>> result = indexer.run()
    >>> print(f"Indexed {result['documents_indexed']} documents to Pinecone")
"""

import logging
from typing import Any

from haystack import Document

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class PineconeHybridIndexingPipeline:
    """Pinecone hybrid (dense + sparse) indexing pipeline.

    Indexes documents with both dense and sparse embeddings into Pinecone,
    enabling hybrid search with native Pinecone sparse vector support.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense semantic embeddings.
        sparse_embedder: Optional component for generating sparse lexical
            embeddings (e.g., SPLADE). None if sparse indexing disabled.
        db: PineconeVectorDB instance for document storage.
        index_name: Name of the Pinecone index to use.
        namespace: Namespace for document isolation within the index.
        batch_size: Number of documents to process per batch.
        show_progress: Whether to display progress bars during upsert.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'pinecone' section with index settings and
                optionally 'sparse' section for sparse embedding configuration.

        Raises:
            ValueError: If required configuration keys are missing or invalid.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> pipeline = PineconeHybridIndexingPipeline("configs/pinecone.yaml")
            >>> pipeline = PineconeHybridIndexingPipeline(
            ...     {
            ...         "pinecone": {"index_name": "my-index"},
            ...         "embeddings": {
            ...             "model": "sentence-transformers/all-MiniLM-L6-v2"
            ...         },
            ...         "sparse": {"model": "naver/splade-cocondenser-ensembledistil"},
            ...     }
            ... )
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.dense_embedder = EmbedderFactory.create_document_embedder(self.config)
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_document_embedder(
                self.config
            )

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config.get("api_key"),
            index_name=pinecone_config.get("index_name"),
            host=pinecone_config.get("host"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "default")
        self.batch_size = pinecone_config.get("batch_size", 100)
        self.show_progress = pinecone_config.get("show_progress", False)

        logger.info("Initialized Pinecone hybrid indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from configured datasource, generates both dense and
        sparse embeddings, and upserts them into Pinecone in batches.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Total number of documents indexed
            - db: Database identifier ("pinecone")
            - index_name: Name of the target Pinecone index

        Raises:
            RuntimeError: If embedding or upsert operations fail.
        """
        logger.info("Starting Pinecone hybrid indexing pipeline")

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
            return {"documents_indexed": 0, "db": "pinecone"}

        embedded_docs = self._embed_documents(documents)
        logger.info("Embedded %d documents", len(embedded_docs))

        try:
            self.db.describe_index(self.index_name)
            logger.info("Using existing Pinecone index: %s", self.index_name)
        except Exception as e:
            logger.warning("Index %s not found: %s", self.index_name, e)

        self.db.upsert(
            documents=embedded_docs,
            index_name=self.index_name,
            namespace=self.namespace,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
        )
        logger.info(
            "Indexed %d documents to Pinecone index %s (namespace: %s)",
            len(embedded_docs),
            self.index_name,
            self.namespace,
        )

        return {
            "documents_indexed": len(embedded_docs),
            "db": "pinecone",
            "index_name": self.index_name,
        }

    def _embed_documents(self, documents: list[Document]) -> list[Document]:
        """Generate dense and sparse embeddings for documents.

        Processes documents through both dense and sparse embedders if
        configured. Dense embeddings capture semantic meaning while
        sparse embeddings capture lexical term importance.

        Args:
            documents: List of Haystack Document objects to embed.

        Returns:
            Documents with embeddings attached to embedding field.

        Note:
            Documents are modified in-place by embedders. Sparse embeddings
            are represented as dictionaries mapping token indices to weights.
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
