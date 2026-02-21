"""Pinecone metadata filtering search pipeline (LangChain).

This module provides the search pipeline for Pinecone vector database with
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

Pinecone Metadata Filtering Capabilities:
    Pinecone supports server-side metadata filtering through its filter syntax:
    - $eq: Exact match for string, numeric, boolean, or list values
    - $ne: Not equal
    - $gt, $gte: Greater than (numeric, strings ordered lexicographically)
    - $lt, $lte: Less than (numeric, strings ordered lexicographically)
    - $in: Value in provided list
    - $nin: Value not in provided list
    - Logical operators: $and, $or

    Filter Example:
    {"$and": [{"category": {"$eq": "technical"}}, {"date": {"$gte": "2024-01-01"}}]}

Pinecone as Metadata Filtered Search Engine:
    Pinecone is ideal for filtered search because:
    - Fully managed infrastructure with sub-100ms query latency
    - Automatic metadata indexing for fast filtering
    - Namespace support for logical data segregation
    - Hybrid search combining dense vectors and sparse BM25
    - Serverless auto-scaling without operational overhead

Configuration:
    pinecone:
      api_key: "your-api-key"  # Pinecone API key (required)
      index_name: "metadata-filtering"  # Target index
      namespace: ""  # Optional namespace

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
    >>> from vectordb.langchain.metadata_filtering.search.pinecone import (
    ...     PineconeMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = PineconeMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning frameworks",
    ...     top_k=10,
    ...     filters={"category": {"$eq": "technical"}},
    ... )
    >>> print(f"Found {len(results['documents'])} documents")

    With RAG generation:
    >>> results = searcher.search("what is deep learning?", top_k=5)
    >>> print(results["answer"])

See Also:
    vectordb.langchain.metadata_filtering.indexing.pinecone: Metadata filtering indexing
    vectordb.langchain.utils.document_filter: Client-side document filtering
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    DocumentFilter,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class PineconeMetadataFilteringSearchPipeline:
    """Pinecone search pipeline for metadata filtering (LangChain).

    This pipeline executes filtered vector search against a Pinecone index,
    applying metadata constraints during query execution and optionally
    generating RAG answers from the filtered results.

    Pinecone metadata filtering search is ideal for:
    - Production deployments requiring managed infrastructure
    - Low-latency queries with complex metadata predicates
    - Auto-scaling workloads without operational overhead
    - Hybrid search combining semantic and keyword relevance

    Attributes:
        config: Loaded configuration dictionary containing pinecone, embedder,
            filters, and optional rag settings.
        embedder: Initialized embedding model instance for query vectorization.
        db: PineconeVectorDB instance for database operations.
        index_name: Name of the Pinecone index to search.
        namespace: Optional namespace for document segregation.
        llm: Optional LLM instance for RAG answer generation.
        filters_config: Client-side filter configuration from config file.

    Example:
        >>> searcher = PineconeMetadataFilteringSearchPipeline("config.yaml")
        >>> results = searcher.search(
        ...     "machine learning", top_k=10, filters={"category": {"$eq": "technical"}}
        ... )
        >>> print(f"Found {len(results['documents'])} documents")
        Found 8 documents

    Configuration Requirements:
        The config file must specify:
        - pinecone.api_key: Pinecone API authentication key (required)
        - pinecone.index_name: Target index name
        - pinecone.namespace: Optional namespace (default: "")
        - embedder: Embedding model configuration
        - filters.conditions: Optional client-side filter definitions
        - rag.llm: Optional LLM for RAG generation
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone metadata filtering search pipeline.

        Loads configuration, initializes the embedding model, connects to Pinecone,
        and sets up optional RAG and client-side filtering.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain pinecone section with
                API key and index details.

        Raises:
            ValueError: If required configuration keys (pinecone.api_key) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            AuthenticationError: If Pinecone API key is invalid.

        Example:
            >>> searcher = PineconeMetadataFilteringSearchPipeline("config.yaml")
            >>> print(searcher.index_name)
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

        # Optional RAG
        self.llm = RAGHelper.create_llm(self.config)

        self.filters_config = self.config.get("filters", {})

        logger.info(
            "Initialized Pinecone metadata filtering search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute metadata filtering search against Pinecone index.

        Embeds the query, executes filtered vector search, applies optional
        client-side filters, and generates RAG answer if configured.

        Args:
            query: Search query text to vectorize and match.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters to apply server-side.
                Uses Pinecone filter syntax: {"field": {"$eq": "value"}}

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents matching query and filters
            - query: Original query string
            - answer: Generated RAG answer if LLM is configured, else None

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If search execution fails.
            ConnectionError: If Pinecone API is unreachable.

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
            ...     filters={
            ...         "category": {"$eq": "technical"},
            ...         "date": {"$gte": "2024-01-01"},
            ...     },
            ... )
            >>> print(
            ...     f"Found: {len(results['documents'])}, "
            ...     f"Answer: {results.get('answer', 'N/A')}"
            ... )
        """
        # Embed query
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Search Pinecone
        documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            namespace=self.namespace,
        )
        logger.info("Retrieved %d documents from Pinecone", len(documents))

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

        # Optional RAG generation
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
