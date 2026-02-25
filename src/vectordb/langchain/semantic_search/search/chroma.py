"""Chroma semantic search pipeline (LangChain).

This module provides the search pipeline for semantic (dense vector) search
using Chroma as the vector database backend. The pipeline embeds queries,
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

Chroma Integration:
    Chroma stores vectors locally and provides fast approximate nearest
    neighbor search. It's ideal for development, testing, and small to
    medium deployments.

    Chroma uses cosine similarity by default for comparing vectors.
    Results are returned with similarity scores indicating how well
    each document matches the query.

Search Parameters:
    query: The search query text to embed and match
    top_k: Number of top results to return (default: 10)
    filters: Optional metadata filters to constrain results

    Filters use Chroma's where clause syntax:
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
    chroma:
      path: "./chroma_data"
      collection_name: "semantic_search"

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"

    rag:
      enabled: true
      model: "llama-3.3-70b-versatile"

Example:
    >>> from vectordb.langchain.semantic_search.search.chroma import (
    ...     ChromaSemanticSearchPipeline,
    ... )
    >>> pipeline = ChromaSemanticSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="What is machine learning?",
    ...     top_k=5,
    ...     filters={"category": "technology"},
    ... )
    >>> for doc in results["documents"]:
    ...     print(f"[{doc.metadata.get('score', 'N/A')}] {doc.page_content[:100]}")

See Also:
    vectordb.langchain.semantic_search.indexing.chroma: Indexing pipeline
    vectordb.langchain.utils.embeddings: Embedding utilities
    vectordb.langchain.utils.rag: RAG generation utilities
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)
from vectordb.utils.chroma_document_converter import ChromaDocumentConverter


logger = logging.getLogger(__name__)


class ChromaSemanticSearchPipeline:
    """Chroma semantic search pipeline (LangChain).

    Implements dense vector similarity search on Chroma collections.
    Queries are embedded and matched against stored document embeddings
    to find semantically similar documents.

    This pipeline is ideal for semantic search applications requiring:
    - Local, persistent vector storage
    - Fast similarity search
    - Optional RAG answer generation
    - Simple deployment configuration

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for query vectorization.
        db: ChromaVectorDB instance for local vector storage.
        collection_name: Name of Chroma collection to search.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> config = {
        ...     "chroma": {
        ...         "path": "./chroma_data",
        ...         "collection_name": "documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = ChromaSemanticSearchPipeline(config)
        >>> results = pipeline.search("neural networks", top_k=10)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize semantic search pipeline from configuration.

        Validates configuration and initializes Chroma connection and
        embedding model for query processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain chroma and embedder sections.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path", "./chroma_data"),
        )

        self.collection_name = chroma_config.get("collection_name", "semantic_search")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Chroma semantic search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search against Chroma collection.

        Embeds the query and performs similarity search to find the most
        semantically similar documents. Optionally generates a RAG answer
        if an LLM is configured.

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of top results to return (default: 10).
            filters: Optional metadata filters to constrain search results.
                Uses Chroma's where clause syntax.

        Returns:
            Dictionary containing:
                - documents: List of Document objects with similarity scores
                - query: Original search query
                - answer: Generated RAG answer (if LLM configured)

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
        # Embed query for similarity search
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        self.db._get_collection(self.collection_name)
        results_dict = self.db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters,
        )
        documents = (
            ChromaDocumentConverter.convert_query_results_to_langchain_documents(
                results_dict
            )
        )
        logger.info("Retrieved %d documents from Chroma", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        # Generate RAG answer if LLM is configured
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
