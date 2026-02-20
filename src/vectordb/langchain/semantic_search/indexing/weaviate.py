"""Weaviate semantic search indexing pipeline (LangChain).

This module provides the indexing pipeline for semantic (dense vector) search
using Weaviate as the vector database backend. The pipeline loads documents,
generates dense embeddings, and indexes them for similarity retrieval.

Semantic Search Architecture:
    Semantic search uses dense vector embeddings to find documents based on
    meaning rather than keyword matching. Documents are embedded into a
    high-dimensional vector space where semantically similar documents
    are close together.

Pipeline Flow:
    1. Load documents from configured data source (TriviaQA, ARC, PopQA, etc.)
    2. Generate dense embeddings using configured embedding model
    3. Create or use existing Weaviate collection
    4. Upsert documents with embeddings and metadata

Weaviate as Vector Store:
    Weaviate is an open-source vector database with a GraphQL interface and
    modular AI integrations. It's ideal for:
    - Knowledge graph applications
    - Semantic search with rich data models
    - Multi-modal data (text, images, etc.)
    - Applications requiring GraphQL query flexibility

    Weaviate supports vector search combined with BM25 hybrid search
    for improved relevance in certain use cases.

Embedding Model Considerations:
    - Same embedder must be used for indexing and search
    - Common choices: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d)
    - Model choice affects retrieval quality and inference speed
    - Dimension must match the collection's vector configuration

Configuration:
    weaviate:
      url: "http://localhost:8080"  # Weaviate server URL
      api_key: null  # Optional API key for authentication
      collection_name: "SemanticSearch"  # Collection/class name

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      device: "cpu"  # or "cuda" for GPU

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.semantic_search.indexing.weaviate import (
    ...     WeaviateSemanticIndexingPipeline,
    ... )
    >>> pipeline = WeaviateSemanticIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    vectordb.langchain.semantic_search.search.weaviate: Search pipeline
    vectordb.langchain.utils.embeddings: Embedding utilities
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class WeaviateSemanticIndexingPipeline:
    """Weaviate indexing pipeline for semantic search (LangChain).

    Loads documents from configured data source, generates dense embeddings,
    and indexes them in a Weaviate collection for similarity retrieval.

    This pipeline is the foundation for semantic search applications. The
    indexed documents can be queried using semantic similarity search to
    find documents that match the meaning of a query.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: WeaviateVectorDB instance for vector storage operations.
        collection_name: Name of Weaviate collection/class for document storage.

    Example:
        >>> config = {
        ...     "weaviate": {
        ...         "url": "http://localhost:8080",
        ...         "collection_name": "Documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"type": "triviaqa"},
        ... }
        >>> pipeline = WeaviateSemanticIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize semantic search indexing pipeline from configuration.

        Validates configuration and initializes Weaviate connection and
        embedding model for document processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain weaviate and embedder sections.

        Raises:
            ValueError: If required configuration is missing.
            KeyError: If weaviate.url is not provided in configuration.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name", "SemanticSearch")

        logger.info("Initialized Weaviate semantic indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs the complete indexing workflow: loads documents from the
        configured data source, generates embeddings, and upserts all
        documents with their vectors to Weaviate.

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

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
