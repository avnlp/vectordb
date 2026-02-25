"""Weaviate parent document retrieval search pipeline (LangChain).

This module implements the search pipeline for parent document retrieval
using Weaviate as the vector database. It retrieves chunks by similarity
and maps them back to their parent documents for context-rich results.

Search Pipeline Flow:
    1. Load configuration and initialize WeaviateVectorDB
    2. Load ParentDocumentStore from disk (required for parent lookup)
    3. For each search query:
        a. Embed query using configured embedder
        b. Search Weaviate for similar chunks (retrieves top_k * 2)
        c. Extract chunk IDs from result metadata
        d. Map chunk IDs to parent documents via ParentDocumentStore
        e. Deduplicate and limit to top_k unique parents
        f. Optionally generate RAG answer using LLM

Parent Retrieval Strategy:
    The pipeline retrieves more chunks than needed (top_k * 2) because:
        - Multiple chunks may belong to the same parent
        - This ensures we have enough unique parents after deduplication
        - Common case: 3 chunks from parent A, 2 from parent B, etc.
    After retrieval, parents are deduplicated and limited to top_k.

Result Structure:
    Returns a dictionary with:
        - parent_documents: List of parent document dictionaries
        - query: Original search query string
        - answer: Optional RAG-generated answer (if LLM configured)

Configuration Requirements:
    Required Weaviate config:
        - url: Weaviate server URL
        - api_key: Optional API key for authentication
        - collection_name: Name of Weaviate collection
    Required parent_store config:
        - store_path: Path to saved ParentDocumentStore pickle file
    Optional config:
        - llm: LLM configuration for RAG answer generation

Example:
    >>> searcher = WeaviateParentDocumentRetrievalSearchPipeline("config.yaml")
    >>> results = searcher.search(
    ...     query="What is machine learning?",
    ...     top_k=5,
    ...     filters={"category": "technology"},
    ... )
    >>> for doc in results["parent_documents"]:
    ...     print(f"Parent: {doc['text'][:200]}...")
    >>> if "answer" in results:
    ...     print(f"Answer: {results['answer']}")
"""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.parent_document_retrieval.parent_store import (
    ParentDocumentStore,
)
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class WeaviateParentDocumentRetrievalSearchPipeline:
    """Weaviate search pipeline for parent document retrieval (LangChain).

    Retrieves chunk embeddings from Weaviate, maps chunks to parent documents
    using ParentDocumentStore, and returns the full parent documents. Supports
    optional RAG answer generation.

    Attributes:
        config: Loaded configuration dictionary
        embedder: Embedding model instance from EmbedderHelper
        db: WeaviateVectorDB instance for vector search
        collection_name: Name of Weaviate collection
        parent_store: ParentDocumentStore for chunk-to-parent mapping
        llm: Optional LLM instance for RAG generation

    Example:
        >>> searcher = WeaviateParentDocumentRetrievalSearchPipeline(
        ...     {
        ...         "weaviate": {
        ...             "url": "http://localhost:8080",
        ...             "collection_name": "my-collection",
        ...         },
        ...         "embedder": {
        ...             "type": "sentence_transformers",
        ...             "model": "all-MiniLM-L6-v2",
        ...         },
        ...         "parent_store": {"store_path": "./cache/parent_store.pkl"},
        ...     }
        ... )
        >>> results = searcher.search("machine learning", top_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Loads configuration, initializes Weaviate connection, sets up embedder,
        and loads the ParentDocumentStore from disk.

        Args:
            config_or_path: Config dict or path to YAML configuration file.
                Must contain weaviate, embedder, and parent_store sections.

        Raises:
            ValueError: If required config keys are missing.
            FileNotFoundError: If parent_store store_path does not exist.

        Example:
            >>> searcher = WeaviateParentDocumentRetrievalSearchPipeline(
            ...     "/path/to/config.yaml"
            ... )
            >>> searcher = WeaviateParentDocumentRetrievalSearchPipeline(
            ...     {
            ...         "weaviate": {
            ...             "url": "http://localhost:8080",
            ...             "collection_name": "docs",
            ...         },
            ...         "embedder": {
            ...             "type": "sentence_transformers",
            ...             "model": "all-MiniLM-L6-v2",
            ...         },
            ...         "parent_store": {"store_path": "./cache/parent_store.pkl"},
            ...     }
            ... )
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        # Initialize embedding model
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize Weaviate connection
        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        # Store collection name
        self.collection_name = weaviate_config.get("collection_name")

        # Load parent document store from disk
        # This is required for chunk-to-parent mapping
        parent_store_path = self.config.get("parent_store", {}).get("store_path")
        if parent_store_path:
            self.parent_store = ParentDocumentStore.load(parent_store_path)
        else:
            # Initialize empty store if no path provided
            self.parent_store = ParentDocumentStore()
            logger.warning("Parent store not loaded from path")

        # Initialize optional LLM for RAG generation
        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Weaviate parent document retrieval search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute parent document retrieval search.

        Performs semantic search on chunks and returns full parent documents.

        Search Strategy:
            1. Embed query text
            2. Search Weaviate for top_k * 2 chunks (extra for deduplication)
            3. Extract chunk IDs from result metadata
            4. Map chunk IDs to parent documents via ParentDocumentStore
            5. Deduplicate parents and limit to top_k
            6. Optionally generate RAG answer

        Args:
            query: Search query text. Will be embedded using configured embedder.
            top_k: Number of unique parent documents to return.
                More chunks are retrieved internally to ensure enough unique parents.
            filters: Optional metadata filters for Weaviate query (where clause).
                Example: {"category": "technology", "year": 2023}

        Returns:
            Dictionary containing:
                - parent_documents: List of parent document dictionaries with keys:
                    - text: Full parent document text
                    - metadata: Document metadata
                    - source_index: Original document index
                - query: Original search query string
                - answer: Optional RAG-generated answer (only if LLM configured)

        Example:
            >>> results = searcher.search(
            ...     query="What is deep learning?", top_k=5, filters={"category": "ai"}
            ... )
            >>> print(f"Found {len(results['parent_documents'])} parents")
            >>> for doc in results["parent_documents"]:
            ...     print(f"Document: {doc['text'][:100]}...")
        """
        # Embed query text for semantic search
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Search Weaviate for similar chunks
        # Retrieve more chunks than needed to account for duplicates
        chunk_documents = self.db.query(
            vector=query_embedding,
            top_k=top_k * 2,
            collection_name=self.collection_name,
            where=filters,
        )
        logger.info("Retrieved %d chunks from Weaviate", len(chunk_documents))

        # Extract chunk IDs from result metadata
        # Chunk IDs were stored as metadata during indexing
        chunk_ids = []
        for doc in chunk_documents:
            if hasattr(doc, "metadata") and "id" in doc.metadata:
                chunk_ids.append(doc.metadata["id"])

        # Map chunk IDs to parent documents
        # ParentDocumentStore handles deduplication internally
        parent_documents = self.parent_store.get_parents_for_chunks(chunk_ids)
        logger.info("Retrieved %d unique parent documents", len(parent_documents))

        # Limit results to requested top_k
        parent_documents = parent_documents[:top_k]

        # Build result dictionary
        result = {
            "parent_documents": parent_documents,
            "query": query,
        }

        # Generate RAG answer if LLM is configured and parents were found
        if self.llm is not None and parent_documents:
            # Convert parent documents to LangChain Document objects
            parent_docs = [
                Document(page_content=doc.get("text", "")) for doc in parent_documents
            ]
            answer = RAGHelper.generate(self.llm, query, parent_docs)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result

    def set_parent_store(self, parent_store: ParentDocumentStore) -> None:
        """Set the parent document store.

        Allows runtime replacement of the parent document store. Useful for
        testing or when loading different parent stores dynamically.

        Args:
            parent_store: ParentDocumentStore instance containing chunk-to-parent
                mappings for this collection.

        Example:
            >>> new_store = ParentDocumentStore.load("./cache/new_store.pkl")
            >>> searcher.set_parent_store(new_store)
        """
        self.parent_store = parent_store
