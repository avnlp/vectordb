"""Pinecone semantic search pipeline (LangChain).

This module provides a standard semantic search pipeline for Pinecone vector
database. Semantic search uses dense vector embeddings to find documents with
similar meaning to the query, enabling conceptual matching beyond keyword overlap.

Semantic Search Process:
    1. Query Embedding: Convert query text to dense vector using embedding model
    2. Vector Search: Find nearest neighbors in Pinecone index
    3. Result Retrieval: Return top-k most similar documents
    4. Optional RAG: Generate answer using retrieved documents as context

Why Semantic Search:
    Traditional keyword search fails when:
    - Query uses synonyms ("automobile" vs "car")
    - Documents use different terminology for same concepts
    - User doesn't know exact terms used in documents

    Semantic search handles these by encoding meaning, not just keywords.

Embedding Consistency:
    Critical: Query and documents must use the SAME embedding model.
    Mixing models (e.g., indexing with MiniLM, querying with OpenAI) will
    produce nonsensical results as vectors exist in different spaces.

Configuration Schema:
    Required:
        pinecone.api_key: Pinecone API authentication
        pinecone.index_name: Target index name
    Optional:
        pinecone.namespace: Document organization namespace
        embedder: Embedding model configuration (must match indexing model)
        rag: Optional LLM configuration for answer generation

Search Parameters:
    - top_k: Number of results to return (default: 10)
    - filters: Metadata filters for pre-filtering (e.g., {"category": "tech"})
    - namespace: Isolated document collection within index

RAG Integration:
    If an LLM is configured, the pipeline can generate answers using retrieved
    documents as context. This combines retrieval accuracy with generation
    fluency for question-answering applications.

Example:
    >>> from vectordb.langchain.semantic_search.search import (
    ...     PineconeSemanticSearchPipeline,
    ... )
    >>> pipeline = PineconeSemanticSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="machine learning applications in healthcare",
    ...     top_k=5,
    ...     filters={"category": "medical"},
    ... )
    >>> for doc in results["documents"]:
    ...     print(f"Result: {doc.page_content[:100]}...")
    >>> if "answer" in results:
    ...     print(f"Answer: {results['answer']}")

See Also:
    - vectordb.langchain.semantic_search.indexing.pinecone: Document indexing
    - vectordb.PineconeVectorDB: Core Pinecone vector database wrapper
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class PineconeSemanticSearchPipeline:
    """Pinecone semantic search pipeline (LangChain).

    Implements semantic document retrieval using dense vector embeddings.
    Embeds queries and retrieves the most semantically similar documents
    from Pinecone. Optionally generates RAG answers using retrieved context.

    Attributes:
        config: Validated configuration dictionary for Pinecone connection,
            embedder settings, and optional LLM configuration.
        embedder: LangChain embedder for query vectorization.
        db: PineconeVectorDB instance for vector search operations.
        index_name: Name of the Pinecone index to search.
        namespace: Pinecone namespace for document organization.
        llm: Optional LangChain LLM for RAG answer generation.

    Design Decisions:
        - Dense retrieval: Uses neural embeddings for semantic matching rather
          than sparse TF-IDF or BM25.
        - Same-model requirement: Query and document embeddings must use the
          same model to be comparable in vector space.
        - Optional RAG: Answer generation is decoupled from retrieval, allowing
          the same pipeline to be used for both search and Q&A.

    Example:
        >>> config = {
        ...     "pinecone": {"api_key": "pc-api-...", "index_name": "docs"},
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = PineconeSemanticSearchPipeline(config)
        >>> results = pipeline.search("neural networks", top_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone semantic search pipeline.

        Loads configuration, initializes the embedder, establishes connection
        to Pinecone, and optionally configures an LLM for RAG.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'pinecone' section with API key and index details.

        Raises:
            ValueError: If required Pinecone configuration (api_key) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Pinecone API.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        # Initialize embedder for query vectorization.
        # Must be the same model used during indexing for valid comparisons.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Establish connection to Pinecone vector database.
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        # Store Pinecone settings for search operations.
        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        # Initialize optional LLM for RAG answer generation.
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Pinecone semantic search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search against the Pinecone index.

        Embeds the query and retrieves the most semantically similar documents
        from Pinecone. Uses cosine similarity (or configured metric) to rank
        results by relevance to the query.

        The search process:
            1. Embed query text using configured embedder
            2. Query Pinecone for nearest neighbors in vector space
            3. Apply optional metadata filters during retrieval
            4. Return top-k most similar documents
            5. Optionally generate RAG answer using retrieved context

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of documents to return. Default is 10.
            filters: Optional metadata filters to apply during retrieval.
                Dictionary mapping field names to filter values.
                Example: {"category": "technology", "year": 2024}

        Returns:
            Dictionary containing:
                - documents: List of Document objects sorted by relevance
                - query: Original query string
                - answer: Generated RAG answer if LLM configured (optional)

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation or Pinecone query fails.

        Example:
            >>> results = pipeline.search(
            ...     query="renewable energy sources",
            ...     top_k=5,
            ...     filters={"category": "environment"},
            ... )
            >>> print(f"Found {len(results['documents'])} relevant documents")
        """
        # Generate query embedding for semantic search.
        # This creates a dense vector representing query meaning.
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Query Pinecone for nearest neighbors in vector space.
        # Returns documents sorted by similarity (highest first).
        documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            namespace=self.namespace,
        )
        logger.info("Retrieved %d documents from Pinecone", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        # Generate RAG answer if LLM is configured.
        # Uses retrieved documents as context for answer generation.
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
