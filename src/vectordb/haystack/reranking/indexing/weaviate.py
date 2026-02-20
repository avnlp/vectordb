"""Weaviate reranking indexing pipeline.

This module provides an indexing pipeline for preparing document collections
in Weaviate for two-stage retrieval with reranking.

Indexing for Reranking:
    Documents are embedded using bi-encoder models and stored in Weaviate.
    These embeddings enable fast vector similarity search during retrieval.
    The cross-encoder reranker processes raw text, so only the initial
    retrieval stage requires vector embeddings in the index.

Pipeline Steps:
    1. Load documents from configured data sources
    2. Generate dense embeddings using bi-encoder models
    3. Create Weaviate collection with vector index
    4. Upsert embedded documents with metadata properties

Weaviate Features:
    - Native vector search with HNSW index
    - Schema-based collections with typed properties
    - Rich metadata filtering with where clauses
    - GraphQL query interface
    - Module system for vectorization and AI integrations

Schema Considerations:
    Collections store document content and metadata as properties,
    with vector embeddings in a dedicated vector index. The schema
    defines which properties are indexed for filtering.
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class WeaviateRerankingIndexingPipeline:
    """Weaviate indexing pipeline for reranking document collections.

    Prepares document collections for two-stage retrieval by generating
    bi-encoder embeddings and storing them in Weaviate. Documents are
    indexed with their embeddings for fast vector search and metadata
    for filtering during retrieval.

    Attributes:
        config: Pipeline configuration dict.
        embedder: Bi-encoder component for document embedding generation.
        dimension: Embedding dimension from embedder configuration.
        db: WeaviateVectorDB instance for collection management.
        collection_name: Name of the Weaviate collection to create/use.

    Example:
        >>> pipeline = WeaviateRerankingIndexingPipeline("weaviate_config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - weaviate: url, collection_name, schema settings
                - embedder: Provider, model, dimensions for bi-encoder
                - dataloader: Dataset source and optional limit

        Raises:
            ValueError: If required config sections are missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)
        self.dimension = EmbedderFactory.get_embedding_dimension(self.embedder)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config.get("url", "http://localhost:8080"),
            collection_name=weaviate_config.get("collection_name", "Reranking"),
        )

        self.collection_name = weaviate_config.get("collection_name", "Reranking")

        logger.info("Initialized Weaviate reranking indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Loads documents from data sources, generates bi-encoder embeddings,
        creates the Weaviate collection with appropriate schema, and
        upserts all documents with embeddings and metadata.

        Returns:
            Dict with 'documents_indexed' count.

        Raises:
            RuntimeError: If embedding generation or Weaviate upsert fails.
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

        self.db.create_collection(
            collection_name=self.collection_name,
        )

        self.db.upsert(documents=embedded_docs)
        num_indexed = len(embedded_docs)
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
