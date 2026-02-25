"""Chroma metadata filtering search pipeline (LangChain).

This module provides the search pipeline for Chroma vector database with
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

Chroma Metadata Filtering Capabilities:
    Chroma supports native metadata filtering through its WHERE clause syntax:
    - $eq: Exact match
    - $ne: Not equal
    - $gt, $gte: Greater than (for numeric metadata)
    - $lt, $lte: Less than (for numeric metadata)
    - $in: Value in list
    - $nin: Value not in list
    - Logical operators: $and, $or

    Filter Example:
    {"$and": [{"category": {"$eq": "technical"}}, {"date": {"$gte": "2024-01-01"}}]}

Chroma as Metadata Filtered Search Engine:
    Chroma is ideal for filtered search because:
    - Local execution with no network latency
    - Native WHERE clause filtering during query
    - Simple deployment without cloud dependencies
    - Fast prototyping and development

Configuration:
    chroma:
      path: "./chroma_data"  # Local directory for persistence
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
    >>> from vectordb.langchain.metadata_filtering.search.chroma import (
    ...     ChromaMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = ChromaMetadataFilteringSearchPipeline("config.yaml")
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
    vectordb.langchain.metadata_filtering.indexing.chroma: Metadata filtering indexing
    vectordb.langchain.utils.document_filter: Client-side document filtering
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    DocumentFilter,
    EmbedderHelper,
    RAGHelper,
)
from vectordb.utils.chroma_document_converter import ChromaDocumentConverter


logger = logging.getLogger(__name__)


class ChromaMetadataFilteringSearchPipeline:
    """Chroma search pipeline for metadata filtering (LangChain).

    This pipeline executes filtered vector search against a Chroma collection,
    applying metadata constraints during query execution and optionally
    generating RAG answers from the filtered results.

    Chroma metadata filtering search is ideal for:
    - Local development and testing with structured filtering
    - Fast prototyping of filtering strategies
    - Small to medium document collections
    - Offline or air-gapped environments

    Attributes:
        config: Loaded configuration dictionary containing chroma, embedder,
            filters, and optional rag settings.
        embedder: Initialized embedding model instance for query vectorization.
        db: ChromaVectorDB instance for database operations.
        collection_name: Name of the Chroma collection to search.
        llm: Optional LLM instance for RAG answer generation.
        filters_config: Client-side filter configuration from config file.

    Example:
        >>> searcher = ChromaMetadataFilteringSearchPipeline("config.yaml")
        >>> results = searcher.search(
        ...     "machine learning", top_k=10, filters={"category": {"$eq": "technical"}}
        ... )
        >>> print(f"Found {len(results['documents'])} documents")
        Found 8 documents

    Configuration Requirements:
        The config file must specify:
        - chroma.path: Local directory for Chroma persistence
        - chroma.collection_name: Target collection name (default: "metadata_filtering")
        - embedder: Embedding model configuration
        - filters.conditions: Optional client-side filter definitions
        - rag.llm: Optional LLM for RAG generation
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma metadata filtering search pipeline.

        Loads configuration, initializes the embedding model, connects to Chroma,
        and sets up optional RAG and client-side filtering.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain chroma section with
                connection details.

        Raises:
            ValueError: If required configuration keys (chroma, embedder) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            RuntimeError: If unable to connect to Chroma database.

        Example:
            >>> searcher = ChromaMetadataFilteringSearchPipeline("config.yaml")
            >>> print(searcher.collection_name)
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
        self.llm = RAGHelper.create_llm(self.config)

        self.filters_config = self.config.get("filters", {})

        logger.info("Initialized Chroma metadata filtering search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute metadata filtering search against Chroma collection.

        Embeds the query, executes filtered vector search, applies optional
        client-side filters, and generates RAG answer if configured.

        Args:
            query: Search query text to vectorize and match.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters to apply server-side.
                Uses Chroma WHERE clause syntax: {"field": {"$eq": "value"}}

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents matching query and filters
            - query: Original query string
            - answer: Generated RAG answer if LLM is configured, else None

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If search execution fails.

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
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        self.db._get_collection(self.collection_name)
        raw_results = self.db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters,
        )
        if isinstance(raw_results, list):
            documents = raw_results
        else:
            documents = (
                ChromaDocumentConverter.convert_query_results_to_langchain_documents(
                    raw_results,
                )
            )
        logger.info("Retrieved %d documents from Chroma", len(documents))

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
