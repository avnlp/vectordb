"""Weaviate hybrid indexing pipeline.

This module provides hybrid indexing capabilities for Weaviate vector database,
combining dense semantic embeddings with Weaviate's native BM25 lexical search.

Weaviate Hybrid Search:
    Unlike other databases that require explicit sparse embeddings, Weaviate
    implements hybrid search by combining:
    1. Dense vector search (user-provided embeddings)
    2. Native BM25 keyword search (automatically computed from text)

    This means no sparse embedder is required for hybrid search in Weaviate.
    The database automatically tokenizes and indexes text for BM25 retrieval.

Sparse Embedding Note:
    While Weaviate doesn't need sparse embeddings for hybrid search, this
    pipeline still supports them for compatibility. If sparse configuration
    is provided, sparse embeddings are generated and stored as document
    metadata, though they are not used in Weaviate's native hybrid search.

Key Features:
    - Native BM25 + vector hybrid search (no sparse embedder needed)
    - Automatic text tokenization and BM25 indexing
    - Configurable alpha weighting between BM25 and vector scores
    - Collection-based organization of documents

Example:
    >>> from vectordb.haystack.hybrid_indexing.indexing.weaviate import (
    ...     WeaviateHybridIndexingPipeline,
    ... )
    >>> indexer = WeaviateHybridIndexingPipeline(
    ...     config_path="configs/weaviate/triviaqa.yaml"
    ... )
    >>> result = indexer.run()
    >>> print(f"Indexed {result['documents_indexed']} documents to Weaviate")
"""

import logging
from typing import Any

from haystack import Document

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class WeaviateHybridIndexingPipeline:
    """Weaviate hybrid indexing pipeline.

    Indexes documents into Weaviate. Weaviate natively supports BM25 search
    alongside vector search for hybrid retrieval, eliminating the need for
    explicit sparse embeddings.

    Attributes:
        config: Loaded and validated configuration dictionary.
        dense_embedder: Component for generating dense semantic embeddings.
        sparse_embedder: Optional sparse embedder (stored but not used by
            Weaviate's native hybrid search).
        db: WeaviateVectorDB instance for document storage.
        collection_name: Name of the Weaviate collection to create/use.
        dimension: Vector dimension for the collection.
        batch_size: Number of documents to upsert per batch.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'weaviate' section with connection settings and
                'embeddings' section for dense embedding configuration.
                Sparse section is optional and not required for Weaviate.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_path is provided but file does not exist.

        Example:
            >>> pipeline = WeaviateHybridIndexingPipeline("configs/weaviate.yaml")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        # Initialize dense embedder for semantic similarity
        self.dense_embedder = EmbedderFactory.create_document_embedder(self.config)

        # Sparse embedder is optional and stored but not used by Weaviate's BM25
        self.sparse_embedder = None
        if "sparse" in self.config:
            self.sparse_embedder = EmbedderFactory.create_sparse_document_embedder(
                self.config
            )

        # Configure Weaviate connection
        weaviate_config = self.config["weaviate"]
        url = weaviate_config.get("url", "http://localhost:8080")
        api_key = weaviate_config.get("api_key")

        config_dict = {"weaviate": {"url": url, "api_key": api_key}}
        self.db = WeaviateVectorDB(config=config_dict)

        self.collection_name = weaviate_config.get("collection_name")
        self.dimension = weaviate_config.get("dimension", 768)
        self.batch_size = weaviate_config.get("batch_size", 100)

        logger.info("Initialized Weaviate hybrid indexing pipeline at %s", url)

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents, generates dense embeddings, creates Weaviate collection,
        and upserts documents in batches. Weaviate automatically handles BM25
        indexing from the document text content.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Total number of documents indexed
            - db: Database identifier ("weaviate")
            - collection_name: Name of the target Weaviate collection

        Raises:
            RuntimeError: If collection creation or upsert operations fail.
        """
        logger.info("Starting Weaviate hybrid indexing pipeline")

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
            return {"documents_indexed": 0, "db": "weaviate"}

        # Generate embeddings (dense required, sparse optional)
        embedded_docs = self._embed_documents(documents)
        logger.info("Embedded %d documents", len(embedded_docs))

        # Create or recreate Weaviate collection
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
        )
        logger.info("Created Weaviate collection: %s", self.collection_name)

        # Upsert documents in batches to manage memory
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
            "Indexed %d documents to Weaviate collection %s",
            len(embedded_docs),
            self.collection_name,
        )

        return {
            "documents_indexed": len(embedded_docs),
            "db": "weaviate",
            "collection_name": self.collection_name,
        }

    def _embed_documents(self, documents: list[Document]) -> list[Document]:
        """Generate dense and optional sparse embeddings for documents.

        Dense embeddings are required for vector search in Weaviate. Sparse
        embeddings are optional and stored as metadata but not used in
        Weaviate's native BM25-based hybrid search.

        Args:
            documents: List of Haystack Document objects to embed.

        Returns:
            Documents with dense embeddings attached. May also have sparse
            embeddings if configured.
        """
        # Generate dense semantic embeddings (required)
        dense_result = self.dense_embedder.run(documents=documents)
        embedded_docs = dense_result["documents"]
        logger.debug("Generated dense embeddings for %d documents", len(embedded_docs))

        # Generate sparse embeddings if configured (optional for Weaviate)
        if self.sparse_embedder:
            sparse_result = self.sparse_embedder.run(documents=embedded_docs)
            embedded_docs = sparse_result["documents"]
            logger.debug(
                "Generated sparse embeddings for %d documents", len(embedded_docs)
            )

        return embedded_docs
