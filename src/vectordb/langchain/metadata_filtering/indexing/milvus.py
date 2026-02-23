"""Milvus metadata filtering indexing pipeline (LangChain).

This module provides the indexing pipeline for Milvus vector database with
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

Milvus Metadata Filtering Capabilities:
    Milvus supports sophisticated metadata (scalar) filtering through its
    expression syntax compatible with boolean and comparison operators:
    - ==, !=: Equality and inequality
    - >, >=, <, <=: Comparison operators for numeric and string fields
    - IN, NOT IN: Set membership
    - LIKE: Pattern matching for strings
    - AND, OR, NOT: Logical operators

    This enables complex queries like:
    "category == 'technical' AND date >= '2024-01-01' AND author IN ['Alice', 'Bob']"

Milvus as Metadata Filtered Retriever:
    Milvus is ideal for metadata-filtered retrieval because:
    - Distributed architecture scales to billions of vectors with metadata
    - Native scalar filtering integrated with vector search
    - Partition key support for efficient data segregation
    - Field indexing for fast metadata predicate evaluation

Configuration:
    milvus:
      host: "localhost"  # Milvus server host
      port: 19530  # Milvus server port
      collection_name: "metadata_filtering"  # Target collection
      dimension: 384  # Embedding vector dimension
      recreate: false  # Whether to drop and recreate collection

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.metadata_filtering.indexing.milvus import (
    ...     MilvusMetadataFilteringIndexingPipeline,
    ... )
    >>> pipeline = MilvusMetadataFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents with metadata")

    Search with metadata filters:
    >>> from vectordb.langchain.metadata_filtering.search.milvus import (
    ...     MilvusMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = MilvusMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning",
    ...     top_k=10,
    ...     filters={"category": "technical", "date": {"$gte": "2024-01-01"}},
    ... )

See Also:
    vectordb.langchain.metadata_filtering.search.milvus: Metadata filtering search
    vectordb.langchain.utils.document_filter: Document filter utilities
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


class MilvusMetadataFilteringIndexingPipeline:
    """Milvus indexing pipeline for metadata filtering (LangChain).

    This pipeline loads documents, generates embeddings, and indexes them
    into a Milvus collection with metadata fields preserved for subsequent
    metadata-filtered search operations.

    Milvus metadata filtering is ideal for:
    - Large-scale production deployments with billions of documents
    - Complex filtering requirements with multiple metadata fields
    - Distributed systems requiring horizontal scaling
    - Hybrid search combining vector similarity and scalar filters

    Attributes:
        config: Loaded configuration dictionary containing milvus, embedder,
            and dataloader settings.
        embedder: Initialized embedding model instance for generating
            dense vector representations.
        db: MilvusVectorDB instance for database operations.
        collection_name: Name of the Milvus collection for indexing.
        dimension: Embedding vector dimension for collection schema.

    Example:
        >>> pipeline = MilvusMetadataFilteringIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents with metadata")
        Indexed 1000 documents with metadata

    Configuration Requirements:
        The config file must specify:
        - milvus.host: Milvus server hostname (default: "localhost")
        - milvus.port: Milvus server port (default: 19530)
        - milvus.collection_name: Target collection name (default: "metadata_filtering")
        - milvus.dimension: Embedding vector dimension (default: 384)
        - milvus.recreate: Whether to drop and recreate collection (default: False)
        - embedder: Embedding model configuration
        - dataloader: Data source configuration with metadata fields

    Note:
        Documents should include metadata fields (category, date, author, etc.)
        to enable effective metadata filtering during search. Milvus supports
        both fixed schema and dynamic field modes for metadata.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus metadata filtering indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the Milvus database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain milvus section with
                connection details.

        Raises:
            ValueError: If required configuration keys (milvus, embedder) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Milvus server.

        Example:
            >>> pipeline = MilvusMetadataFilteringIndexingPipeline("config.yaml")
            >>> print(pipeline.collection_name)
            metadata_filtering
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get(
            "collection_name", "metadata_filtering"
        )
        self.dimension = milvus_config.get("dimension", 384)

        logger.info(
            "Initialized Milvus metadata filtering indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete metadata filtering indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and inserts all documents with their metadata into the Milvus collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.
            ValueError: If document loading returns invalid data.
            ConnectionError: If Milvus connection is lost during indexing.

        Pipeline Steps:
            1. Load documents with metadata from configured data source
            2. Generate embeddings for all documents using embedder
            3. Create collection (drop and recreate if recreate=True)
            4. Insert documents with embeddings and metadata to Milvus
            5. Return count of indexed documents

        Example:
            >>> result = pipeline.run()
            >>> print(f"Success: {result['documents_indexed']} documents")
            Success: 1000 documents
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

        recreate = self.config.get("milvus", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            recreate=recreate,
        )

        self.db.insert_documents(
            documents=docs,
            collection_name=self.collection_name,
        )
        num_indexed = len(docs)
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
