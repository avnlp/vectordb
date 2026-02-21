"""Milvus metadata filtering search pipeline (LangChain).

This module provides the search pipeline for Milvus vector database with
metadata filtering support. Metadata filtering combines vector similarity search
with structured metadata constraints to retrieve precisely targeted documents.

Why Metadata Filtering:
    - Vector search finds semantically similar content but cannot filter by
      structured properties like dates, categories, or author IDs
    - Metadata filtering applies boolean predicates (equals, range, contains)
      to narrow results before or after vector similarity computation
    - Essential for production RAG: "find documents about ML published in 2024"

Search Pipeline Architecture:
    1. Query Embedding: Convert query text to vector representation
    2. Filtered Vector Search: Execute similarity search with metadata constraints
    3. Post-Processing: Apply additional client-side filters if configured
    4. RAG Generation: Generate answer using filtered documents (optional)

Milvus Metadata Filtering Capabilities:
    Milvus supports sophisticated metadata (scalar) filtering through its
    expression syntax:
    - ==, !=: Equality and inequality
    - >, >=, <, <=: Comparison operators for numeric and string fields
    - IN, NOT IN: Set membership
    - LIKE: Pattern matching for strings
    - AND, OR, NOT: Logical operators

    Filter Example:
    "category == 'technical' AND date >= '2024-01-01' AND author IN ['Alice', 'Bob']"

Milvus as Metadata Filtered Search Engine:
    Milvus is ideal for filtered search because:
    - Distributed architecture scales to billions of vectors
    - Native scalar filtering integrated with vector search
    - Partition keys for efficient data segregation
    - Field indexing for fast metadata predicate evaluation
    - GPU acceleration for large-scale searches

Configuration:
    milvus:
      host: "localhost"  # Milvus server host
      port: 19530  # Milvus server port
      collection_name: "metadata_filtering"  # Target collection

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    filters:
      conditions:
        - field: "category"
          value: "technical"
          operator: "equals"

    rag:
      llm: "openai"  # Optional RAG configuration

Example:
    >>> from vectordb.langchain.metadata_filtering.search.milvus import (
    ...     MilvusMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = MilvusMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning frameworks",
    ...     top_k=10,
    ...     filters={"category": "technical", "date": {"$gte": "2024-01-01"}},
    ... )
    >>> print(f"Found {len(results['documents'])} documents")

    With RAG generation:
    >>> results = searcher.search("what is deep learning?", top_k=5)
    >>> print(results["answer"])

See Also:
    vectordb.langchain.metadata_filtering.indexing.milvus: Metadata filtering indexing
    vectordb.langchain.utils.document_filter: Client-side document filtering
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    DocumentFilter,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class MilvusMetadataFilteringSearchPipeline:
    """Milvus search pipeline for metadata filtering (LangChain).

    This pipeline executes filtered vector search against a Milvus collection,
    applying metadata constraints during query execution and optionally
    generating RAG answers from the filtered results.

    Milvus metadata filtering search is ideal for:
    - Large-scale production deployments
    - Complex filtering with multiple metadata fields
    - Distributed systems requiring horizontal scaling
    - High-throughput search applications

    Attributes:
        config: Loaded configuration dictionary containing milvus, embedder,
            filters, and optional rag settings.
        embedder: Initialized embedding model instance for query vectorization.
        db: MilvusVectorDB instance for database operations.
        collection_name: Name of the Milvus collection to search.
        llm: Optional LLM instance for RAG answer generation.
        filters_config: Client-side filter configuration from config file.

    Example:
        >>> searcher = MilvusMetadataFilteringSearchPipeline("config.yaml")
        >>> results = searcher.search(
        ...     "machine learning", top_k=10, filters={"category": "technical"}
        ... )
        >>> print(f"Found {len(results['documents'])} documents")
        Found 8 documents

    Configuration Requirements:
        The config file must specify:
        - milvus.host: Milvus server hostname (default: "localhost")
        - milvus.port: Milvus server port (default: 19530)
        - milvus.collection_name: Target collection name (default: "metadata_filtering")
        - embedder: Embedding model configuration
        - filters.conditions: Optional client-side filter definitions
        - rag.llm: Optional LLM for RAG generation
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus metadata filtering search pipeline.

        Loads configuration, initializes the embedding model, connects to Milvus,
        and sets up optional RAG and client-side filtering.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain milvus section with
                connection details.

        Raises:
            ValueError: If required configuration keys (milvus, embedder) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Milvus server.

        Example:
            >>> searcher = MilvusMetadataFilteringSearchPipeline("config.yaml")
            >>> print(searcher.collection_name)
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
        self.llm = RAGHelper.create_llm(self.config)

        self.filters_config = self.config.get("filters", {})

        logger.info("Initialized Milvus metadata filtering search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute metadata filtering search against Milvus collection.

        Embeds the query, executes filtered vector search, applies optional
        client-side filters, and generates RAG answer if configured.

        Args:
            query: Search query text to vectorize and match.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters to apply server-side.
                Uses Milvus expression syntax: {"category": "technical"}

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents matching query and filters
            - query: Original query string
            - answer: Generated RAG answer if LLM is configured, else None

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If search execution fails.
            ConnectionError: If Milvus connection is lost during search.

        Search Flow:
            1. Embed query text using configured embedder
            2. Execute vector search with server-side metadata filters
            3. Apply client-side filters if configured
            4. Generate RAG answer using LLM if configured
            5. Return documents and optional answer

        Example:
            >>> results = searcher.search(
            ...     "neural networks",
            ...     top_k=5,
            ...     filters={"category": "technical", "date": {"$gte": "2024-01-01"}},
            ... )
            >>> print(
            ...     f"Found: {len(results['documents'])}, "
            ...     f"Answer: {results.get('answer', 'N/A')}"
            ... )
        """
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d documents from Milvus", len(documents))

        # Apply metadata filtering if configured
        if self.filters_config:
            documents = self._apply_filters(documents)
            logger.info(
                "Applied metadata filters: %d documents remaining", len(documents)
            )

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result

    def _apply_filters(self, documents: list[Any]) -> list[Any]:
        """Apply configured client-side metadata filters to documents.

        Filters documents based on conditions defined in config filters.conditions.
        Supports operators: equals, not_equals, greater_than, less_than, contains.

        Args:
            documents: List of documents to filter.

        Returns:
            Filtered list of documents matching all conditions.

        Example:
            Configured with:
            filters:
              conditions:
                - field: "category"
                  value: "technical"
                  operator: "equals"

            Only documents where metadata["category"] == "technical" are returned.
        """
        filtered = documents
        for filter_def in self.filters_config.get("conditions", []):
            key = filter_def.get("field")
            value = filter_def.get("value")
            operator = filter_def.get("operator", "equals")

            if key and value is not None:
                filtered = DocumentFilter.filter_by_metadata(
                    filtered,
                    key=key,
                    value=value,
                    operator=operator,
                )
        return filtered
