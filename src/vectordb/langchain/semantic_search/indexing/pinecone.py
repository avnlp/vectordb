"""Pinecone semantic search indexing pipeline (LangChain).

This module provides a standard indexing pipeline for Pinecone vector database
optimized for semantic search. Semantic search uses dense vector embeddings to
find documents with similar meaning, enabling conceptual matching beyond
keyword overlap.

Semantic Search vs Keyword Search:
    - Keyword search: Matches exact terms (e.g., "car" matches "car" but not
      "automobile")
    - Semantic search: Matches by meaning (e.g., "car" matches "automobile",
      "vehicle")

    This is achieved by embedding documents into a high-dimensional vector space
    where semantically similar content is positioned close together.

Indexing Pipeline:
    1. Load configuration and validate Pinecone settings
    2. Initialize embedder for dense vector generation
    3. Connect to Pinecone using API key
    4. Load documents from configured data source
    5. Generate embeddings for all documents
    6. Create or recreate Pinecone index
    7. Upsert documents with embeddings to namespace

Embedding Models:
    The choice of embedding model significantly impacts search quality:
    - all-MiniLM-L6-v2: Fast, good quality, 384 dimensions (default)
    - all-mpnet-base-v2: Higher quality, 768 dimensions, slower
    - OpenAI text-embedding-3-small: Excellent quality, 1536 dimensions

    The dimension must match the Pinecone index configuration.

Configuration Schema:
    Required:
        pinecone.api_key: Pinecone API authentication
        pinecone.index_name: Target index name
    Optional:
        pinecone.namespace: Document organization namespace (default: "")
        pinecone.dimension: Vector dimension matching embedder (default: 384)
        pinecone.metric: Distance metric (default: "cosine")
        pinecone.recreate: Whether to recreate index (default: False)
        embedder: Embedding model configuration
        dataloader: Data source configuration

Distance Metrics:
    - cosine: Best for semantic similarity (default)
    - euclidean: Good for absolute distance measures
    - dotproduct: Efficient for normalized vectors

Example:
    >>> from vectordb.langchain.semantic_search.indexing import (
    ...     PineconeSemanticIndexingPipeline,
    ... )
    >>> pipeline = PineconeSemanticIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents for semantic search")

See Also:
    - vectordb.langchain.semantic_search.search.pinecone: Semantic search
    - vectordb.PineconeVectorDB: Core Pinecone vector database wrapper
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)
from vectordb.utils.pinecone_document_converter import PineconeDocumentConverter


logger = logging.getLogger(__name__)


class PineconeSemanticIndexingPipeline:
    """Pinecone indexing pipeline for semantic search (LangChain).

    Prepares document collections for semantic retrieval by indexing documents
    with their dense embeddings. Uses embedding models to capture semantic
    meaning, enabling conceptual search beyond keyword matching.

    Attributes:
        config: Validated configuration dictionary containing Pinecone settings,
            embedder configuration, and data source details.
        embedder: LangChain embedder instance for generating document vectors.
        db: PineconeVectorDB instance for vector store operations.
        index_name: Name of the Pinecone index for documents.
        namespace: Pinecone namespace for document organization.
        dimension: Vector dimension matching the embedder output.

    Design Decisions:
        - Dense embeddings: Uses neural network-based embeddings that capture
          semantic meaning rather than sparse TF-IDF vectors.
        - Cosine similarity: Default metric optimized for semantic similarity
          tasks where vector direction matters more than magnitude.
        - Namespace support: Enables logical document separation within a
          single index for multi-tenant or multi-collection scenarios.

    Example:
        >>> config = {
        ...     "pinecone": {
        ...         "api_key": "pc-api-...",
        ...         "index_name": "semantic-docs",
        ...         "dimension": 384,
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"dataset": "triviaqa"},
        ... }
        >>> pipeline = PineconeSemanticIndexingPipeline(config)
        >>> result = pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone semantic indexing pipeline.

        Loads configuration, validates Pinecone-specific settings, initializes
        the embedder, and establishes connection to Pinecone.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'pinecone' section with API key and index details.

        Raises:
            ValueError: If required Pinecone configuration (api_key) is missing
                or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Pinecone API.

        Configuration Schema:
            pinecone:
                api_key: Pinecone API key (required)
                index_name: Name for the index
                namespace: Namespace for documents (default: "")
                dimension: Vector dimension (default: 384)
                metric: Distance metric (default: "cosine")
                recreate: Whether to recreate index if exists (default: False)
            embedder: Embedder configuration dict
            dataloader: Data source configuration dict
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        # Initialize embedder for dense vector generation.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize Pinecone connection with API key from config.
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        # Store Pinecone settings for pipeline operations.
        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone semantic indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        creates the Pinecone index, and upserts documents. Returns statistics
        about the operation.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Create Pinecone index with appropriate settings
            4. Upsert documents to Pinecone index

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or Pinecone upsert fails.
            ValueError: If no documents are found in the data source.

        Example:
            >>> result = pipeline.run()
            >>> assert result["documents_indexed"] > 0
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

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        # Upsert documents
        upsert_data = PineconeDocumentConverter.prepare_langchain_documents_for_upsert(
            docs, embeddings
        )
        num_indexed = self.db.upsert(
            data=upsert_data,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
