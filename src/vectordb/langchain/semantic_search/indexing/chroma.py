"""Chroma semantic search indexing pipeline (LangChain).

This module provides the indexing pipeline for semantic (dense vector) search
using Chroma as the vector database backend. The pipeline loads documents,
generates dense embeddings, and indexes them for similarity retrieval.

Semantic Search Architecture:
    Semantic search uses dense vector embeddings to find documents based on
    meaning rather than keyword matching. Documents are embedded into a
    high-dimensional vector space where semantically similar documents
    are close together.

Pipeline Flow:
    1. Load documents from configured data source (TriviaQA, ARC, PopQA, etc.)
    2. Generate dense embeddings using configured embedding model
    3. Create or recreate Chroma collection
    4. Upsert documents with embeddings and metadata

Chroma as Vector Store:
    Chroma is an embedded vector database that stores vectors locally on disk.
    It's ideal for:
    - Development and testing environments
    - Prototyping and experimentation
    - Small to medium document collections
    - Offline or air-gapped deployments

    For production with large datasets, consider Pinecone, Milvus, or Qdrant.

Embedding Model Considerations:
    - Same embedder must be used for indexing and search
    - Common choices: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d)
    - Model choice affects retrieval quality and inference speed
    - Dimension must match the vector store's expected size

    Configuration:
    chroma:
      path: "./chroma_data"  # Directory for local storage
      collection_name: "documents"  # Collection name
      recreate: false  # Whether to recreate collection
    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      device: "cpu"  # or "cuda" for GPU

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.semantic_search.indexing.chroma import (
    ...     ChromaSemanticIndexingPipeline,
    ... )
    >>> pipeline = ChromaSemanticIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    vectordb.langchain.semantic_search.search.chroma: Search pipeline
    vectordb.langchain.utils.embeddings: Embedding utilities
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class ChromaSemanticIndexingPipeline:
    """Chroma indexing pipeline for semantic search (LangChain).

    Loads documents from configured data source, generates dense embeddings,
    and indexes them in a local Chroma collection for similarity retrieval.

    This pipeline is the foundation for semantic search applications. The
    indexed documents can be queried using semantic similarity search to
    find documents that match the meaning of a query.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: ChromaVectorDB instance for local vector storage.
        collection_name: Name of Chroma collection for document storage.

    Example:
        >>> config = {
        ...     "chroma": {
        ...         "path": "./chroma_data",
        ...         "collection_name": "documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"type": "triviaqa"},
        ... }
        >>> pipeline = ChromaSemanticIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize semantic search indexing pipeline from configuration.

        Validates configuration and initializes Chroma connection and
        embedding model for document processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain chroma and embedder sections.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path", "./chroma_data"),
        )

        self.collection_name = chroma_config.get("collection_name", "semantic_search")

        logger.info("Initialized Chroma semantic indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs the complete indexing workflow: loads documents from the
        configured data source, generates embeddings, creates the Chroma
        collection, and upserts all documents with their vectors.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed

        Example:
            >>> result = pipeline.run()
            >>> print(f"Successfully indexed {result['documents_indexed']} documents")
        """
        # Load documents with optional limit
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

        # Generate embeddings for all documents
        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d documents", len(docs))

        # Create or recreate collection based on configuration
        recreate = self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            name=self.collection_name,
            recreate=recreate,
        )

        # Upsert documents with embeddings to Chroma
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
