"""Qdrant semantic search pipeline (LangChain).

This module provides the search pipeline for semantic (dense vector) search
using Qdrant as the vector database backend. The pipeline embeds queries,
performs similarity search, and optionally generates RAG answers.

Semantic Search Architecture:
    Semantic search finds documents based on meaning similarity rather than
    keyword matching. The process works as follows:

    1. Query Embedding: Convert the search query to a dense vector using
       the same embedding model used during indexing
    2. Similarity Search: Query the vector database for documents with
       embeddings closest to the query embedding
    3. Result Formatting: Convert database results to LangChain Documents
    4. Optional RAG: Generate an answer using retrieved documents if LLM
       is configured

Qdrant Integration:
    Qdrant provides high-performance approximate nearest neighbor search
    with rich support for metadata filtering and payload storage.

    Qdrant uses HNSW indexing and supports complex filter expressions
    combining multiple conditions with AND/OR logic.

Search Parameters:
    query: The search query text to embed and match
    top_k: Number of top results to return (default: 10)
    filters: Optional metadata filters to constrain results

    Filters use Qdrant's filter syntax:
        {"category": "technology", "year": {"$gte": 2020}}

Results Format:
    Returns a dictionary containing:
        - documents: List of LangChain Document objects
        - query: Original query string
        - answer: Generated RAG answer (if LLM configured)

    Each Document includes:
        - page_content: Document text
        - metadata: Source metadata and similarity score

Configuration:
    qdrant:
      url: "http://localhost:6333"
      api_key: null
      collection_name: "semantic_search"

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    rag:
      enabled: true
      model: "llama-3.3-70b-versatile"

Example:
    >>> from vectordb.langchain.semantic_search.search.qdrant import (
    ...     QdrantSemanticSearchPipeline,
    ... )
    >>> pipeline = QdrantSemanticSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="What is machine learning?",
    ...     top_k=5,
    ...     filters={"category": "technology"},
    ... )
    >>> for doc in results["documents"]:
    ...     print(f"[{doc.metadata.get('score', 'N/A')}] {doc.page_content[:100]}")

See Also:
    vectordb.langchain.semantic_search.indexing.qdrant: Indexing pipeline
    vectordb.langchain.utils.embeddings: Embedding utilities
    vectordb.langchain.utils.rag: RAG generation utilities
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    HaystackToLangchainConverter,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class QdrantSemanticSearchPipeline:
    """Qdrant semantic search pipeline (LangChain).

    Implements dense vector similarity search on Qdrant collections.
    Queries are embedded and matched against stored document embeddings
    to find semantically similar documents.

    This pipeline is ideal for semantic search applications requiring:
    - Rich metadata filtering capabilities
    - Self-hosted deployment options
    - High-performance vector search
    - Optional RAG answer generation

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for query vectorization.
        db: QdrantVectorDB instance for vector storage operations.
        collection_name: Name of Qdrant collection to search.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> config = {
        ...     "qdrant": {
        ...         "url": "http://localhost:6333",
        ...         "collection_name": "documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = QdrantSemanticSearchPipeline(config)
        >>> results = pipeline.search("neural networks", top_k=10)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize semantic search pipeline from configuration.

        Validates configuration and initializes Qdrant connection and
        embedding model for query processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain qdrant and embedder sections.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name", "semantic_search")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Qdrant semantic search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search against Qdrant collection.

        Embeds the query and performs similarity search to find the most
        semantically similar documents. Optionally generates a RAG answer
        if an LLM is configured.

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of top results to return (default: 10).
            filters: Optional metadata filters to constrain search results.
                Uses Qdrant filter syntax.

        Returns:
            Dictionary containing:
                - documents: List of Document objects with similarity scores
                - query: Original search query
                - answer: Generated RAG answer (if LLM configured)

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation or Qdrant query fails.

        Example:
            >>> results = pipeline.search(
            ...     query="What is deep learning?",
            ...     top_k=5,
            ...     filters={"category": "ai"},
            ... )
            >>> print(f"Found {len(results['documents'])} documents")
            >>> for doc in results["documents"]:
            ...     print(f"Score: {doc.metadata.get('score')}")
        """
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        documents = self.db.search(
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        documents = HaystackToLangchainConverter.convert(documents)
        logger.info("Retrieved %d documents from Qdrant", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
