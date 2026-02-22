"""Weaviate metadata filtering search pipeline (LangChain).

This module provides the search pipeline for Weaviate vector database with
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
    3. Document Conversion: Convert Weaviate results to LangChain Documents
    4. Post-Processing: Apply additional client-side filters if configured
    5. RAG Generation: Generate answer using filtered documents (optional)

Weaviate Metadata Filtering Capabilities:
    Weaviate supports sophisticated filtering through GraphQL Where filters:
    - Equal, NotEqual: Exact value matching
    - GreaterThan, GreaterThanEqual, LessThan, LessThanEqual: Range queries
    - Like: Pattern matching with wildcards
    - ContainsAny, ContainsAll: Array membership tests
    - And, Or: Logical operators for complex conditions

    Filter Example:
    {operator: And, operands: [
        {path: ["category"], operator: Equal, valueText: "technical"},
        {path: ["date"], operator: GreaterThanEqual, valueDate: "2024-01-01"}
    ]}

Weaviate as Metadata Filtered Search Engine:
    Weaviate is ideal for filtered search because:
    - GraphQL-native API with rich filtering capabilities
    - Schema-first approach with strict type validation
    - Multi-modal support (text, images, audio) with unified filtering
    - Vector search with configurable ANN algorithms
    - Built-in vectorization modules for automatic embedding

Configuration:
    weaviate:
      url: "http://localhost:8080"  # Weaviate server URL (required)
      api_key: null  # Optional API key for authentication
      collection_name: "MetadataFiltering"  # Target collection

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
    >>> from vectordb.langchain.metadata_filtering.search.weaviate import (
    ...     WeaviateMetadataFilteringSearchPipeline,
    ... )
    >>> searcher = WeaviateMetadataFilteringSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     "machine learning frameworks", top_k=10, filters={"category": "technical"}
    ... )
    >>> print(f"Found {len(results['documents'])} documents")

    With RAG generation:
    >>> results = searcher.search("what is deep learning?", top_k=5)
    >>> print(results["answer"])

See Also:
    vectordb.langchain.metadata_filtering.indexing.weaviate: Metadata filtering indexing
    vectordb.langchain.utils.document_filter: Client-side document filtering
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    DocumentFilter,
    EmbedderHelper,
    RAGHelper,
)
from vectordb.utils.weaviate_document_converter import WeaviateDocumentConverter


logger = logging.getLogger(__name__)


class WeaviateMetadataFilteringSearchPipeline:
    """Weaviate search pipeline for metadata filtering (LangChain).

    This pipeline executes filtered vector search against a Weaviate collection,
    applying metadata constraints during query execution and optionally
    generating RAG answers from the filtered results.

    Weaviate metadata filtering search is ideal for:
    - GraphQL-based applications with complex data relationships
    - Multi-modal search requiring unified filtering
    - Schema-strict environments needing type validation
    - Knowledge graph integration with semantic search

    Attributes:
        config: Loaded configuration dictionary containing weaviate, embedder,
            filters, and optional rag settings.
        embedder: Initialized embedding model instance for query vectorization.
        db: WeaviateVectorDB instance for database operations.
        collection_name: Name of the Weaviate collection to search.
        llm: Optional LLM instance for RAG answer generation.
        filters_config: Client-side filter configuration from config file.

    Example:
        >>> searcher = WeaviateMetadataFilteringSearchPipeline("config.yaml")
        >>> results = searcher.search(
        ...     "machine learning", top_k=10, filters={"category": "technical"}
        ... )
        >>> print(f"Found {len(results['documents'])} documents")
        Found 8 documents

    Configuration Requirements:
        The config file must specify:
        - weaviate.url: Weaviate server URL (required)
        - weaviate.api_key: Optional authentication key
        - weaviate.collection_name: Target collection name
            (default: "MetadataFiltering")
        - embedder: Embedding model configuration
        - filters.conditions: Optional client-side filter definitions
        - rag.llm: Optional LLM for RAG generation
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Weaviate metadata filtering search pipeline.

        Loads configuration, initializes the embedding model, connects to Weaviate,
        and sets up optional RAG and client-side filtering.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file. Must contain weaviate section with
                connection details.

        Raises:
            ValueError: If required configuration keys (weaviate.url) are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Weaviate server.

        Example:
            >>> searcher = WeaviateMetadataFilteringSearchPipeline("config.yaml")
            >>> print(searcher.collection_name)
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
        self.db._select_collection(self.collection_name)
        self.llm = RAGHelper.create_llm(self.config)

        self.filters_config = self.config.get("filters", {})

        logger.info(
            "Initialized Weaviate metadata filtering search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute metadata filtering search against Weaviate collection.

        Embeds the query, executes filtered vector search, applies optional
        client-side filters, and generates RAG answer if configured.

        Args:
            query: Search query text to vectorize and match.
            top_k: Maximum number of results to return (default: 10).
            filters: Optional metadata filters to apply server-side.
                Uses Weaviate filter syntax compatible with GraphQL Where.

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents matching query and filters
            - query: Original query string
            - answer: Generated RAG answer if LLM is configured, else None

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If search execution fails.
            ConnectionError: If Weaviate connection is lost during search.

        Search Flow:
            1. Embed query text using configured embedder
            2. Execute vector search with server-side metadata filters
            3. Convert Weaviate results to LangChain Documents
            4. Apply client-side filters if configured
            5. Generate RAG answer using LLM if configured
            6. Return documents and optional answer

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

        raw_results = self.db.query(
            vector=query_embedding,
            limit=top_k,
            filters=filters,
        )
        if isinstance(raw_results, list):
            documents = raw_results
        else:
            documents = (
                WeaviateDocumentConverter.convert_query_results_to_langchain_documents(
                    raw_results,
                )
            )
        logger.info("Retrieved %d documents from Weaviate", len(documents))

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
