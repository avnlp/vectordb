"""Pinecone metadata filtering indexing pipeline (LangChain).

This module provides the indexing pipeline for Pinecone vector database with
metadata filtering support. Metadata filtering combines vector similarity search
with structured metadata constraints to retrieve precisely targeted documents.

Why Metadata Filtering:
    - Vector search finds semantically similar content but cannot filter by
      structured properties like dates, categories, or author IDs
    - Metadata filtering applies boolean predicates (equals, range, contains)
      to narrow results before or after vector similarity computation
    - Essential for production RAG: "find documents about ML published in 2024"

Filtering Pipeline Architecture:
    1. Indexing: Store documents with embeddings AND metadata fields (this module)
    2. Search: Apply metadata filters during vector query execution
    3. RAG: Generate answers from filtered, semantically relevant documents

Pinecone Metadata Filtering Capabilities:
    Pinecone supports server-side metadata filtering through its filter syntax:
    - $eq: Exact match for string, numeric, boolean, or list values
    - $ne: Not equal
    - $gt, $gte: Greater than (numeric, strings ordered lexicographically)
    - $lt, $lte: Less than (numeric, strings ordered lexicographically)
    - $in: Value in provided list
    - $nin: Value not in provided list
    - Logical operators: $and, $or

    This enables complex queries like:
    {"$and": [{"category": {"$eq": "technical"}}, {"date": {"$gte": "2024-01-01"}}]}

Pinecone as Metadata Filtered Retriever:
    Pinecone is ideal for metadata-filtered retrieval because:
    - Fully managed serverless or pod-based infrastructure
    - Sub-100ms query latency with metadata filtering at scale
    - Automatic indexing of metadata fields for fast filtering
    - Namespace support for logical data segregation
    - Hybrid search combining dense vectors and sparse BM25

Configuration:
    pinecone:
      api_key: "your-api-key"  # Pinecone API key (required)
      index_name: "metadata-filtering"  # Target index
      dimension: 384  # Vector dimension
      metric: "cosine"  # Distance metric
      namespace: ""  # Optional namespace
      recreate: false  # Whether to recreate index

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.metadata_filtering.indexing.pinecone import (
    ...     PineconeMetadataFilteringIndexingPipeline,
    ... )
    >>> pipeline = PineconeMetadataFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents with metadata")

    Search with metadata filters:
    >>> from vectordb.langchain.metadata_filtering.search.pinecone import (
    ...     PineconeMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = PineconeMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning", top_k=10, filters={"category": {"$eq": "technical"}}
    ... )

See Also:
    vectordb.langchain.metadata_filtering.search.pinecone: Metadata filtering search
    vectordb.langchain.utils.document_filter: Document filter utilities
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


class PineconeMetadataFilteringIndexingPipeline:
    """Pinecone indexing pipeline for metadata filtering (LangChain).

    This pipeline loads documents, generates embeddings, and indexes them
    into a Pinecone index with metadata fields preserved for subsequent
    metadata-filtered search operations.

    Pinecone metadata filtering is ideal for:
    - Production deployments requiring managed infrastructure
    - Low-latency queries with complex metadata predicates
    - Hybrid search combining semantic and keyword relevance
    - Auto-scaling workloads without operational overhead

    Attributes:
        config: Loaded configuration dictionary containing pinecone, embedder,
            and dataloader settings.
        embedder: Initialized embedding model instance for generating
            dense vector representations.
        db: PineconeVectorDB instance for database operations.
        index_name: Name of the Pinecone index for indexing.
        namespace: Optional namespace for document segregation.
        dimension: Dimension of the embedding vectors.

    Example:
        >>> pipeline = PineconeMetadataFilteringIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents with metadata")
        Indexed 10000 documents with metadata

    Configuration Requirements:
        The config file must specify:
        - pinecone.api_key: Pinecone API authentication key (required)
        - pinecone.index_name: Target index name
        - pinecone.dimension: Vector dimension (default: 384)
        - pinecone.metric: Distance metric (default: "cosine")
        - pinecone.namespace: Optional namespace (default: "")
        - pinecone.recreate: Whether to recreate index (default: False)
        - embedder: Embedding model configuration
        - dataloader: Data source configuration with metadata fields

    Note:
        Documents should include metadata fields (category, date, author, etc.)
        to enable effective metadata filtering during search. Pinecone metadata
        has a 40KB size limit per record for free tier.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone metadata filtering indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the Pinecone database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain pinecone section with
                API key and index details.

        Raises:
            ValueError: If required configuration keys (pinecone.api_key) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            AuthenticationError: If Pinecone API key is invalid.

        Example:
            >>> pipeline = PineconeMetadataFilteringIndexingPipeline("config.yaml")
            >>> print(pipeline.index_name)
            metadata-filtering
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info(
            "Initialized Pinecone metadata filtering indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete metadata filtering indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        creates the Pinecone index if needed, and upserts all documents with
        their metadata into the index.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.
            ValueError: If document loading returns invalid data.
            ConnectionError: If Pinecone API is unreachable.

        Pipeline Steps:
            1. Load documents with metadata from configured data source
            2. Generate embeddings for all documents using embedder
            3. Create or recreate Pinecone index with specified dimension
            4. Upsert documents with embeddings and metadata to Pinecone
            5. Return count of indexed documents

        Example:
            >>> result = pipeline.run()
            >>> print(f"Success: {result['documents_indexed']} documents")
            Success: 10000 documents
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
        logger.info("Created Pinecone index: %s", self.index_name)

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
