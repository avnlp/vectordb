"""Milvus semantic search pipeline (LangChain).

This module provides the search pipeline for semantic (dense vector) search
using Milvus as the vector database backend. The pipeline embeds queries,
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

Milvus Integration:
    Milvus provides high-performance approximate nearest neighbor search
    with support for complex filtering and distributed deployment.

    Milvus supports multiple distance metrics (cosine, L2, IP) and index
    types (HNSW, IVF_FLAT) optimized for different performance requirements.

Search Parameters:
    query: The search query text to embed and match
    top_k: Number of top results to return (default: 10)
    filters: Optional metadata filters to constrain results

    Filters use Milvus's expression syntax:
        {"category": "technology", "year": { "$gte": 2020 }}

Results Format:
    Returns a dictionary containing:
        - documents: List of LangChain Document objects
        - query: Original query string
        - answer: Generated RAG answer (if LLM configured)

    Each Document includes:
        - page_content: Document text
        - metadata: Source metadata and similarity score

Configuration:
    milvus:
      host: "localhost"
      port: 19530
      collection_name: "semantic_search"

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    rag:
      enabled: true
      model: "llama-3.3-70b-versatile"

Example:
    >>> from vectordb.langchain.semantic_search.search.milvus import (
    ...     MilvusSemanticSearchPipeline,
    ... )
    >>> pipeline = MilvusSemanticSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="What is machine learning?",
    ...     top_k=5,
    ...     filters={"category": "technology"},
    ... )
    >>> for doc in results["documents"]:
    ...     print(f"[{doc.metadata.get('score', 'N/A')}] {doc.page_content[:100]}")

See Also:
    vectordb.langchain.semantic_search.indexing.milvus: Indexing pipeline
    vectordb.langchain.utils.embeddings: Embedding utilities
    vectordb.langchain.utils.rag: RAG generation utilities
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class MilvusSemanticSearchPipeline:
    """Milvus semantic search pipeline (LangChain).

    Implements dense vector similarity search on Milvus collections.
    Queries are embedded and matched against stored document embeddings
    to find semantically similar documents.

    This pipeline is ideal for semantic search applications requiring:
    - High-performance vector search at scale
    - Distributed deployment capabilities
    - Complex metadata filtering
    - Optional RAG answer generation

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for query vectorization.
        db: MilvusVectorDB instance for vector storage operations.
        collection_name: Name of Milvus collection to search.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> config = {
        ...     "milvus": {
        ...         "host": "localhost",
        ...         "port": 19530,
        ...         "collection_name": "documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = MilvusSemanticSearchPipeline(config)
        >>> results = pipeline.search("neural networks", top_k=10)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize semantic search pipeline from configuration.

        Validates configuration and initializes Milvus connection and
        embedding model for query processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain milvus and embedder sections.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get("collection_name", "semantic_search")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Milvus semantic search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search against Milvus collection.

        Embeds the query and performs similarity search to find the most
        semantically similar documents. Optionally generates a RAG answer
        if an LLM is configured.

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of top results to return (default: 10).
            filters: Optional metadata filters to constrain search results.
                Uses Milvus expression syntax.

        Returns:
            Dictionary containing:
                - documents: List of Document objects with similarity scores
                - query: Original search query
                - answer: Generated RAG answer (if LLM configured)

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation or Milvus query fails.

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
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d documents from Milvus", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
