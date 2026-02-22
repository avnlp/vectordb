"""Qdrant semantic search indexing pipeline (LangChain).

This module provides the indexing pipeline for semantic (dense vector) search
using Qdrant as the vector database backend. The pipeline loads documents,
generates dense embeddings, and indexes them for similarity retrieval.

Semantic Search Architecture:
    Semantic search uses dense vector embeddings to find documents based on
    meaning rather than keyword matching. Documents are embedded into a
    high-dimensional vector space where semantically similar documents
    are close together.

Pipeline Flow:
    1. Load documents from configured data source (TriviaQA, ARC, PopQA, etc.)
    2. Generate dense embeddings using configured embedding model
    3. Create or use existing Qdrant collection
    4. Upsert documents with embeddings and metadata

Qdrant as Vector Store:
    Qdrant is an open-source vector database with a focus on filtering
    support and ease of use. It's ideal for:
    - Development and production environments
    - Applications requiring rich metadata filtering
    - Self-hosted deployments
    - Real-time vector search with payload filtering

    Qdrant uses HNSW indexing for approximate nearest neighbor search
    and supports complex filtering on document metadata.

Embedding Model Considerations:
    - Same embedder must be used for indexing and search
    - Common choices: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d)
    - Model choice affects retrieval quality and inference speed
    - Dimension must match the collection's vector configuration

Configuration:
    qdrant:
      url: "http://localhost:6333"  # Qdrant server URL
      api_key: null  # Optional API key for authentication
      collection_name: "semantic_search"  # Collection name

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      device: "cpu"  # or "cuda" for GPU

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.semantic_search.indexing.qdrant import (
    ...     QdrantSemanticIndexingPipeline,
    ... )
    >>> pipeline = QdrantSemanticIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    vectordb.langchain.semantic_search.search.qdrant: Search pipeline
    vectordb.langchain.utils.embeddings: Embedding utilities
"""

import logging
import uuid
from typing import Any

from qdrant_client.models import PointStruct

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class QdrantSemanticIndexingPipeline:
    """Qdrant indexing pipeline for semantic search (LangChain).

    Loads documents from configured data source, generates dense embeddings,
    and indexes them in a Qdrant collection for similarity retrieval.

    This pipeline is the foundation for semantic search applications. The
    indexed documents can be queried using semantic similarity search to
    find documents that match the meaning of a query.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: QdrantVectorDB instance for vector storage operations.
        collection_name: Name of Qdrant collection for document storage.

    Example:
        >>> config = {
        ...     "qdrant": {
        ...         "url": "http://localhost:6333",
        ...         "collection_name": "documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"type": "triviaqa"},
        ... }
        >>> pipeline = QdrantSemanticIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize semantic search indexing pipeline from configuration.

        Validates configuration and initializes Qdrant connection and
        embedding model for document processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain qdrant and embedder sections.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.collection_name = qdrant_config.get("collection_name", "semantic_search")

        self.db = QdrantVectorDB(
            config={
                "qdrant": {
                    "url": qdrant_config.get("url", "http://localhost:6333"),
                    "api_key": qdrant_config.get("api_key"),
                    "collection_name": self.collection_name,
                }
            }
        )

        logger.info("Initialized Qdrant semantic indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs the complete indexing workflow: loads documents from the
        configured data source, generates embeddings, and upserts all
        documents with their vectors to Qdrant.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed

        Example:
            >>> result = pipeline.run()
            >>> print(f"Successfully indexed {result['documents_indexed']} documents")
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_langchain()
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d documents", len(docs))

        dimension = len(embeddings[0]) if embeddings else 0
        self.db.create_collection(dimension=dimension)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "page_content": doc.page_content,
                    **doc.metadata,
                },
            )
            for doc, embedding in zip(docs, embeddings)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.db.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
                wait=False,
            )
        logger.info("Indexed %d documents to Qdrant", len(docs))

        return {"documents_indexed": len(docs)}
