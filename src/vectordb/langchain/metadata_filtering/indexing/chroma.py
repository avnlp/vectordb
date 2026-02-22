"""Chroma metadata filtering indexing pipeline (LangChain).

This module provides the indexing pipeline for Chroma vector database with
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

Chroma Metadata Filtering Capabilities:
    Chroma supports native metadata filtering through its WHERE clause syntax:
    - $eq: Exact match
    - $ne: Not equal
    - $gt, $gte: Greater than (for numeric metadata)
    - $lt, $lte: Less than (for numeric metadata)
    - $in: Value in list
    - $nin: Value not in list
    - Logical operators: $and, $or

    This enables complex queries like:
    {"category": {"$eq": "technical"}, "date": {"$gte": "2024-01-01"}}

Chroma as Metadata Filtered Retriever:
    Chroma is ideal for metadata-filtered retrieval because:
    - Local persistent storage with configurable path
    - Native WHERE clause filtering during query execution
    - No network latency for filter evaluation
    - Efficient for small to medium collections (< 1M docs)

Configuration:
    chroma:
      path: "./chroma_data"  # Local directory for persistence
      collection_name: "metadata_filtering"  # Target collection
      recreate: false  # Whether to delete and recreate collection

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.metadata_filtering.indexing.chroma import (
    ...     ChromaMetadataFilteringIndexingPipeline,
    ... )
    >>> pipeline = ChromaMetadataFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents with metadata")

    Search with metadata filters:
    >>> from vectordb.langchain.metadata_filtering.search.chroma import (
    ...     ChromaMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = ChromaMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning", top_k=10, filters={"category": {"$eq": "technical"}}
    ... )

See Also:
    vectordb.langchain.metadata_filtering.search.chroma: Metadata filtering search
    vectordb.langchain.utils.document_filter: Document filter utilities
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


class ChromaMetadataFilteringIndexingPipeline:
    """Chroma indexing pipeline for metadata filtering (LangChain).

    This pipeline loads documents, generates embeddings, and indexes them
    into a local Chroma collection with metadata fields preserved for
    subsequent metadata-filtered search operations.

    Chroma metadata filtering is ideal for:
    - Local development and testing with structured filtering
    - Small to medium document collections with rich metadata
    - Offline or air-gapped environments requiring precise filtering
    - Prototyping filtering strategies before cloud deployment

    Attributes:
        config: Loaded configuration dictionary containing chroma, embedder,
            and dataloader settings.
        embedder: Initialized embedding model instance for generating
            dense vector representations.
        db: ChromaVectorDB instance for database operations.
        collection_name: Name of the Chroma collection for indexing.

    Example:
        >>> pipeline = ChromaMetadataFilteringIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents with metadata")
        Indexed 100 documents with metadata

    Configuration Requirements:
        The config file must specify:
        - chroma.path: Local directory for Chroma persistence
        - chroma.collection_name: Target collection name (default: "metadata_filtering")
        - chroma.recreate: Whether to delete existing collection (default: False)
        - embedder: Embedding model configuration
        - dataloader: Data source configuration with metadata fields

    Note:
        Documents should include metadata fields (category, date, author, etc.)
        to enable effective metadata filtering during search.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma metadata filtering indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the local Chroma database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain chroma section with
                connection details.

        Raises:
            ValueError: If required configuration keys (chroma, embedder) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Example:
            >>> pipeline = ChromaMetadataFilteringIndexingPipeline("config.yaml")
            >>> print(pipeline.collection_name)
            metadata_filtering
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path", "./chroma_data"),
        )

        self.collection_name = chroma_config.get(
            "collection_name", "metadata_filtering"
        )

        logger.info(
            "Initialized Chroma metadata filtering indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete metadata filtering indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts all documents with their metadata into the Chroma collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.
            ValueError: If document loading returns invalid data.

        Pipeline Steps:
            1. Load documents with metadata from configured data source
            2. Generate embeddings for all documents using embedder
            3. Delete existing collection if recreate=True
            4. Upsert documents with embeddings and metadata to Chroma
            5. Return count of indexed documents

        Example:
            >>> result = pipeline.run()
            >>> print(f"Success: {result['documents_indexed']} documents")
            Success: 100 documents
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

        recreate = self.config.get("chroma", {}).get("recreate", False)
        if recreate:
            self.db.delete_collection(self.collection_name)
            logger.info("Recreated Chroma collection: %s", self.collection_name)

        data = {
            "ids": [doc.metadata.get("id", str(i)) for i, doc in enumerate(docs)],
            "documents": [doc.page_content for doc in docs],
            "metadatas": [doc.metadata for doc in docs],
            "embeddings": embeddings,
        }
        self.db.upsert(data)
        logger.info("Indexed %d documents to Chroma", len(docs))

        return {"documents_indexed": len(docs)}
