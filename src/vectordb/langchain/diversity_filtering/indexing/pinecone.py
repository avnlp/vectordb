"""Pinecone diversity filtering indexing pipeline (LangChain).

This module provides an indexing pipeline for Pinecone vector database optimized
for diversity filtering retrieval. Diversity filtering ensures search results
are not only relevant but also cover different aspects of the query, reducing
redundancy and improving information coverage.

Diversity Filtering Strategy:
    Unlike standard semantic search that returns the k most similar documents,
    diversity filtering post-processes results to select documents that are:
    - Relevant to the query (high similarity)
    - Diverse from each other (low inter-document similarity)

    This is achieved through two methods:
    1. Threshold-based: Filters out documents too similar to already-selected ones
    2. Clustering-based: Groups documents into clusters, samples from each cluster

Indexing Requirements:
    The indexing pipeline stores document embeddings that will be used during
    search for both query matching AND inter-document similarity calculations.
    No special metadata is required - diversity is computed at search time.

Pipeline Flow:
    1. Load configuration and validate Pinecone settings
    2. Initialize embedder for dense vector generation
    3. Connect to Pinecone using API key
    4. Load documents from configured data source
    5. Generate embeddings for all documents
    6. Create or recreate Pinecone index
    7. Upsert documents with embeddings to namespace

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

Example:
    >>> from vectordb.langchain.diversity_filtering.indexing import (
    ...     PineconeDiversityFilteringIndexingPipeline,
    ... )
    >>> pipeline = PineconeDiversityFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(
    ...     f"Indexed {result['documents_indexed']} documents for diversity filtering"
    ... )

See Also:
    - vectordb.langchain.diversity_filtering.search.pinecone: Search with diversity
    - vectordb.utils.diversification_helper: Diversity algorithm implementations
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class PineconeDiversityFilteringIndexingPipeline:
    """Pinecone indexing pipeline for diversity filtering search (LangChain).

    Prepares document collections for diversity-aware retrieval by indexing
    documents with their dense embeddings. The embeddings enable both query
    matching and inter-document similarity calculations during search-time
    diversity filtering.

    Attributes:
        config: Validated configuration dictionary containing Pinecone settings,
            embedder configuration, and data source details.
        embedder: LangChain embedder instance for generating document vectors.
        db: PineconeVectorDB instance for vector store operations.
        index_name: Name of the Pinecone index for documents.
        namespace: Pinecone namespace for document organization.
        dimension: Vector dimension matching the embedder output.

    Design Decisions:
        - Standard dense embeddings: Uses the same embeddings for query matching
          and diversity calculation, avoiding the need for separate vectors.
        - Search-time diversity: Diversity filtering happens during search, not
          indexing, allowing flexible diversity parameters per query.
        - Namespace isolation: Supports multi-tenant scenarios via namespaces.

    Example:
        >>> config = {
        ...     "pinecone": {
        ...         "api_key": "pc-api-...",
        ...         "index_name": "diverse-docs",
        ...         "dimension": 384,
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"dataset": "triviaqa"},
        ... }
        >>> pipeline = PineconeDiversityFilteringIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone diversity filtering indexing pipeline.

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
        # Same embeddings used for query matching and diversity calculation.
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

        logger.info(
            "Initialized Pinecone diversity filtering indexing pipeline (LangChain)"
        )

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
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
