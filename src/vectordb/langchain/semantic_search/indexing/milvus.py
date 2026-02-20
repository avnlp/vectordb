"""Milvus semantic search indexing pipeline (LangChain).

This module provides the indexing pipeline for semantic (dense vector) search
using Milvus as the vector database backend. The pipeline loads documents,
generates dense embeddings, and indexes them for similarity retrieval.

Semantic Search Architecture:
    Semantic search uses dense vector embeddings to find documents based on
    meaning rather than keyword matching. Documents are embedded into a
    high-dimensional vector space where semantically similar documents
    are close together.

Pipeline Flow:
    1. Load documents from configured data source (TriviaQA, ARC, PopQA, etc.)
    2. Generate dense embeddings using configured embedding model
    3. Create or use existing Milvus collection
    4. Upsert documents with embeddings and metadata

Milvus as Vector Store:
    Milvus is a cloud-native vector database designed for high-performance
    similarity search at scale. It's ideal for:
    - Large-scale document collections (millions to billions)
    - High-throughput search applications
    - Hybrid search combining vector and scalar filtering
    - Production deployments requiring horizontal scalability

    Milvus supports multiple index types (IVF_FLAT, HNSW, etc.) optimized
    for different performance/recall tradeoffs.

Embedding Model Considerations:
    - Same embedder must be used for indexing and search
    - Common choices: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d)
    - Model choice affects retrieval quality and inference speed
    - Dimension must match the collection's vector field size

Configuration:
    milvus:
      host: "localhost"  # Milvus server host
      port: 19530  # Milvus server port
      collection_name: "semantic_search"  # Collection name
      dimension: 384  # Vector dimension

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      device: "cpu"  # or "cuda" for GPU

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.semantic_search.indexing.milvus import (
    ...     MilvusSemanticIndexingPipeline,
    ... )
    >>> pipeline = MilvusSemanticIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

See Also:
    vectordb.langchain.semantic_search.search.milvus: Search pipeline
    vectordb.langchain.utils.embeddings: Embedding utilities
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class MilvusSemanticIndexingPipeline:
    """Milvus indexing pipeline for semantic search (LangChain).

    Loads documents from configured data source, generates dense embeddings,
    and indexes them in a Milvus collection for similarity retrieval.

    This pipeline is the foundation for semantic search applications. The
    indexed documents can be queried using semantic similarity search to
    find documents that match the meaning of a query.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: MilvusVectorDB instance for vector storage operations.
        collection_name: Name of Milvus collection for document storage.
        dimension: Vector dimension matching the embedder output.

    Example:
        >>> config = {
        ...     "milvus": {
        ...         "host": "localhost",
        ...         "port": 19530,
        ...         "collection_name": "documents",
        ...         "dimension": 384,
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"type": "triviaqa"},
        ... }
        >>> pipeline = MilvusSemanticIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize semantic search indexing pipeline from configuration.

        Validates configuration and initializes Milvus connection and
        embedding model for document processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain milvus and embedder sections.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get("collection_name", "semantic_search")
        self.dimension = milvus_config.get("dimension", 384)

        logger.info("Initialized Milvus semantic indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs the complete indexing workflow: loads documents from the
        configured data source, generates embeddings, and upserts all
        documents with their vectors to Milvus.

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
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
