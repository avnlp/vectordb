"""Chroma reranking indexing pipeline.

This module provides an indexing pipeline for preparing document collections
in Chroma for two-stage retrieval with reranking.

Indexing for Reranking:
    Documents are embedded using bi-encoder models and stored in Chroma.
    These embeddings enable fast vector similarity search during the first
    stage of reranking. The cross-encoder used in the second stage operates
    on raw text content, not embeddings.

Pipeline Steps:
    1. Load documents from configured data sources
    2. Generate dense embeddings using bi-encoder models
    3. Create or recreate Chroma collection with HNSW index
    4. Upsert embedded documents with metadata

Chroma Characteristics:
    - Local or client-server deployment options
    - HNSW index for approximate nearest neighbor search
    - Simple API with minimal configuration
    - Good for development, prototyping, and small-to-medium datasets

Bi-Encoder Selection:
    The bi-encoder model determines embedding quality for retrieval.
    Popular choices include sentence-transformers models (all-MiniLM-L6-v2,
    all-mpnet-base-v2) or OpenAI embeddings. The dimension must match
    the collection configuration.
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class ChromaRerankingIndexingPipeline:
    """Chroma indexing pipeline for reranking document collections.

    Prepares document collections for two-stage retrieval by generating
    bi-encoder embeddings and storing them in Chroma. The indexed embeddings
    enable fast approximate nearest neighbor search during the retrieval
    phase of reranking.

    Attributes:
        config: Pipeline configuration dict.
        embedder: Bi-encoder component for document embedding generation.
        dimension: Embedding dimension from the embedder model.
        db: ChromaVectorDB instance for collection management.
        collection_name: Name of the Chroma collection to create/use.

    Example:
        >>> pipeline = ChromaRerankingIndexingPipeline("chroma_config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Successfully indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - chroma: host, port, collection_name settings
                - embedder: Provider, model, dimensions, batch_size
                - dataloader: Dataset source and optional document limit

        Raises:
            ValueError: If required config sections are missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)
        self.dimension = EmbedderFactory.get_embedding_dimension(self.embedder)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            host=chroma_config.get("host", "localhost"),
            port=chroma_config.get("port", 8000),
            collection_name=chroma_config.get("collection_name", "reranking"),
        )

        self.collection_name = chroma_config.get("collection_name", "reranking")

        logger.info("Initialized Chroma reranking indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Loads documents from the configured data source, generates bi-encoder
        embeddings, creates or recreates the Chroma collection, and upserts
        all documents with their embeddings and metadata.

        Returns:
            Dict with 'documents_indexed' count indicating success.

        Raises:
            RuntimeError: If embedding generation or Chroma upsert fails.
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

        recreate = self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            name=self.collection_name,
            dimension=self.dimension,
            recreate=recreate,
        )

        self.db.upsert(documents=embedded_docs)
        num_indexed = len(embedded_docs)
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
