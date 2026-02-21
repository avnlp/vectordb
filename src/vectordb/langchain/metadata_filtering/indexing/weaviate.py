"""Weaviate metadata filtering indexing pipeline (LangChain).

This module provides the indexing pipeline for Weaviate vector database with
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

Weaviate Metadata Filtering Capabilities:
    Weaviate supports sophisticated filtering through GraphQL Where filters:
    - Equal, NotEqual: Exact value matching
    - GreaterThan, GreaterThanEqual, LessThan, LessThanEqual: Range queries
    - Like: Pattern matching with wildcards
    - ContainsAny, ContainsAll: Array membership tests
    - And, Or: Logical operators for complex conditions

    This enables complex queries like:
    {path: ["category"], operator: Equal, valueText: "technical"}
    AND
    {path: ["date"], operator: GreaterThanEqual, valueDate: "2024-01-01"}

Weaviate as Metadata Filtered Retriever:
    Weaviate is ideal for metadata-filtered retrieval because:
    - GraphQL-native API with rich filtering capabilities
    - Schema-first approach with strict type validation
    - Multi-modal support (text, images, audio) with unified filtering
    - Vector search with configurable ANN algorithms (HNSW, flat)
    - Built-in vectorization modules for automatic embedding

Configuration:
    weaviate:
      url: "http://localhost:8080"  # Weaviate server URL (required)
      api_key: null  # Optional API key for authentication
      collection_name: "MetadataFiltering"  # Target collection
      recreate: false  # Whether to delete and recreate collection

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.metadata_filtering.indexing.weaviate import (
    ...     WeaviateMetadataFilteringIndexingPipeline,
    ... )
    >>> pipeline = WeaviateMetadataFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents with metadata")

    Search with metadata filters:
    >>> from vectordb.langchain.metadata_filtering.search.weaviate import (
    ...     WeaviateMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = WeaviateMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning", top_k=10, filters={"category": "technical"}
    ... )

See Also:
    vectordb.langchain.metadata_filtering.search.weaviate: Metadata filtering search
    vectordb.langchain.utils.document_filter: Document filter utilities
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


class WeaviateMetadataFilteringIndexingPipeline:
    """Weaviate indexing pipeline for metadata filtering (LangChain).

    This pipeline loads documents, generates embeddings, and indexes them
    into a Weaviate collection with metadata fields preserved for subsequent
    metadata-filtered search operations.

    Weaviate metadata filtering is ideal for:
    - GraphQL-based applications with complex data relationships
    - Multi-modal search requiring unified filtering
    - Schema-strict environments needing type validation
    - Knowledge graph integration with semantic search

    Attributes:
        config: Loaded configuration dictionary containing weaviate, embedder,
            and dataloader settings.
        embedder: Initialized embedding model instance for generating
            dense vector representations.
        db: WeaviateVectorDB instance for database operations.
        collection_name: Name of the Weaviate collection for indexing.

    Example:
        >>> pipeline = WeaviateMetadataFilteringIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents with metadata")
        Indexed 2500 documents with metadata

    Configuration Requirements:
        The config file must specify:
        - weaviate.url: Weaviate server URL (required)
        - weaviate.api_key: Optional authentication key
        - weaviate.collection_name: Target collection name
            (default: "MetadataFiltering")
        - weaviate.recreate: Whether to delete existing collection (default: False)
        - embedder: Embedding model configuration
        - dataloader: Data source configuration with metadata fields

    Note:
        Documents should include metadata fields (category, date, author, etc.)
        to enable effective metadata filtering during search. Weaviate's
        schema-first approach may require explicit property definitions.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate metadata filtering indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the Weaviate database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain weaviate section with
                connection details.

        Raises:
            ValueError: If required configuration keys (weaviate.url) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Weaviate server.

        Example:
            >>> pipeline = WeaviateMetadataFilteringIndexingPipeline("config.yaml")
            >>> print(pipeline.collection_name)
            MetadataFiltering
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get(
            "collection_name", "MetadataFiltering"
        )

        logger.info(
            "Initialized Weaviate metadata filtering indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete metadata filtering indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts all documents with their metadata into the Weaviate collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.
            ValueError: If document loading returns invalid data.
            ConnectionError: If Weaviate connection is lost during indexing.

        Pipeline Steps:
            1. Load documents with metadata from configured data source
            2. Generate embeddings for all documents using embedder
            3. Delete existing collection if recreate=True
            4. Upsert documents with embeddings and metadata to Weaviate
            5. Return count of indexed documents

        Example:
            >>> result = pipeline.run()
            >>> print(f"Success: {result['documents_indexed']} documents")
            Success: 2500 documents
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

        recreate = self.config.get("weaviate", {}).get("recreate", False)
        if recreate:
            self.db.delete_collection(self.collection_name)
            logger.info("Recreated Weaviate collection: %s", self.collection_name)

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {"documents_indexed": num_indexed}
