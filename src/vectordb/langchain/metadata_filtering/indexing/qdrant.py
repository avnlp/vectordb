"""Qdrant metadata filtering indexing pipeline (LangChain).

This module provides the indexing pipeline for Qdrant vector database with
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

Qdrant Metadata Filtering Capabilities:
    Qdrant supports advanced payload filtering through its filter DSL:
    - match: Exact value match for keywords, integers, or booleans
    - range: Numeric or datetime range queries with gt, gte, lt, lte
    - geo_bounding_box, geo_radius: Geographic filtering
    - values_count: Filter by number of values in array field
    - is_empty, is_null: Check for field existence or null values
    - must, must_not, should: Logical operators for complex conditions

    This enables complex queries like:
    {"must": [{"key": "category", "match": {"value": "technical"}},
              {"key": "date", "range": {"gte": "2024-01-01"}}]}

Qdrant as Metadata Filtered Retriever:
    Qdrant is ideal for metadata-filtered retrieval because:
    - Open-source with on-premise or managed cloud deployment options
    - Native payload indexing for fast filter evaluation
    - HNSW index with configurable parameters for speed/accuracy trade-off
    - Support for complex nested metadata structures
    - Built-in filtering during vector search (no post-filtering needed)

Configuration:
    qdrant:
      url: "http://localhost:6333"  # Qdrant server URL
      api_key: null  # Optional API key for authentication
      collection_name: "metadata_filtering"  # Target collection
      recreate: false  # Whether to delete and recreate collection

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.metadata_filtering.indexing.qdrant import (
    ...     QdrantMetadataFilteringIndexingPipeline,
    ... )
    >>> pipeline = QdrantMetadataFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents with metadata")

    Search with metadata filters:
    >>> from vectordb.langchain.metadata_filtering.search.qdrant import (
    ...     QdrantMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = QdrantMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning",
    ...     top_k=10,
    ...     filters={"must": [{"key": "category", "match": {"value": "technical"}}]},
    ... )

See Also:
    vectordb.langchain.metadata_filtering.search.qdrant: Metadata filtering search
    vectordb.langchain.utils.document_filter: Document filter utilities
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


class QdrantMetadataFilteringIndexingPipeline:
    """Qdrant indexing pipeline for metadata filtering (LangChain).

    This pipeline loads documents, generates embeddings, and indexes them
    into a Qdrant collection with metadata fields preserved for subsequent
    metadata-filtered search operations.

    Qdrant metadata filtering is ideal for:
    - Open-source deployments requiring full control
    - Complex nested metadata structures
    - Geographic filtering alongside semantic search
    - Cost-effective self-hosted solutions

    Attributes:
        config: Loaded configuration dictionary containing qdrant, embedder,
            and dataloader settings.
        embedder: Initialized embedding model instance for generating
            dense vector representations.
        db: QdrantVectorDB instance for database operations.
        collection_name: Name of the Qdrant collection for indexing.

    Example:
        >>> pipeline = QdrantMetadataFilteringIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents with metadata")
        Indexed 5000 documents with metadata

    Configuration Requirements:
        The config file must specify:
        - qdrant.url: Qdrant server URL (default: "http://localhost:6333")
        - qdrant.api_key: Optional authentication key
        - qdrant.collection_name: Target collection name (default: "metadata_filtering")
        - qdrant.recreate: Whether to delete existing collection (default: False)
        - embedder: Embedding model configuration
        - dataloader: Data source configuration with metadata fields

    Note:
        Documents should include metadata fields (category, date, author, etc.)
        to enable effective metadata filtering during search. Qdrant supports
        nested JSON objects in payloads for complex metadata.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Qdrant metadata filtering indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the Qdrant database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain qdrant section with
                connection details.

        Raises:
            ValueError: If required configuration keys (qdrant) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Qdrant server.

        Example:
            >>> pipeline = QdrantMetadataFilteringIndexingPipeline("config.yaml")
            >>> print(pipeline.collection_name)
            metadata_filtering
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.collection_name = qdrant_config.get(
            "collection_name", "metadata_filtering"
        )

        self.db = QdrantVectorDB(
            config={
                "qdrant": {
                    "url": qdrant_config.get("url", "http://localhost:6333"),
                    "api_key": qdrant_config.get("api_key"),
                    "collection_name": self.collection_name,
                }
            }
        )

        logger.info(
            "Initialized Qdrant metadata filtering indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute the complete metadata filtering indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts all documents with their metadata into the Qdrant collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.
            ValueError: If document loading returns invalid data.
            ConnectionError: If Qdrant connection is lost during indexing.

        Pipeline Steps:
            1. Load documents with metadata from configured data source
            2. Generate embeddings for all documents using embedder
            3. Delete existing collection if recreate=True
            4. Upsert documents with embeddings and metadata to Qdrant
            5. Return count of indexed documents

        Example:
            >>> result = pipeline.run()
            >>> print(f"Success: {result['documents_indexed']} documents")
            Success: 5000 documents
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

        recreate = self.config.get("qdrant", {}).get("recreate", False)
        if recreate:
            self.db.client.delete_collection(self.collection_name)
            logger.info("Recreated Qdrant collection: %s", self.collection_name)

        dimension = len(embeddings[0]) if embeddings else 0
        self.db.create_collection(dimension)

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
