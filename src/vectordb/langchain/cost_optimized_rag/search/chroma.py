"""Chroma cost-optimized RAG search pipeline for LangChain.

This module implements a cost-optimized search pipeline for Chroma vector
database. Due to Chroma's limited sparse vector support, this pipeline
primarily uses dense semantic search with optional metadata-based filtering.

Cost Optimization:
    - Single dense query embedding: One API call for semantic search
    - Local execution: Chroma runs locally, eliminating hosting costs
    - No sparse overhead: Simplified search without RRF fusion
    - Optional LLM: RAG generation can be disabled

Chroma Limitations:
    Chroma has limited native sparse vector support. This pipeline focuses
    on dense semantic search. For hybrid capabilities, sparse embeddings
    are stored as metadata but not used in fusion.

Search Strategy:
    1. Generate dense query embedding via API
    2. Execute dense semantic search in Chroma
    3. Optionally generate RAG answer

Note:
    For full hybrid search with RRF fusion, consider using Pinecone,
    Qdrant, Weaviate, or Milvus which have native sparse support.

Example:
    >>> pipeline = ChromaCostOptimizedRAGSearchPipeline("config.yaml")
    >>> result = pipeline.search(
    ...     "deep learning frameworks", top_k=10, filters={"category": "software"}
    ... )
    >>> for doc in result["documents"]:
    ...     print(doc["text"][:200])
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    SparseEmbedder,
)
from vectordb.utils.chroma_document_converter import ChromaDocumentConverter


logger = logging.getLogger(__name__)


class ChromaCostOptimizedRAGSearchPipeline:
    """Chroma search pipeline for cost-optimized RAG using LangChain.

    This pipeline implements semantic search for Chroma vector database.
    Due to Chroma's architecture, it focuses on dense vector search with
    optional metadata filtering rather than hybrid fusion.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator (unused in search).
        db: ChromaVectorDB client instance.
        collection_name: Name of the Chroma collection.
        llm: Optional language model for RAG generation.
        search_config: Search-specific configuration.
        rrf_k: RRF parameter (not used in Chroma, kept for API consistency).

    Note:
        This pipeline does not implement RRF fusion due to Chroma's
        limited sparse vector support. For hybrid search, use other
        vector databases like Pinecone or Qdrant.

    Example:
        >>> pipeline = ChromaCostOptimizedRAGSearchPipeline("config.yaml")
        >>> result = pipeline.search("python programming", top_k=5)
        >>> for doc in result["documents"]:
        ...     print(doc["text"][:100])
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain chroma section with persist_directory
                and collection_name.
                Must contain embedding section with provider and model.
                Optionally contains llm section for RAG generation.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.

        Note:
            Sparse embedder is initialized but not used in search due to
            Chroma's architecture limitations.
        """
        # Load and validate configuration
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        # Initialize embedders (sparse initialized for consistency but unused)
        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        # Configure Chroma connection (local storage)
        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_directory=chroma_config.get("persist_directory", "./chroma_db"),
        )

        self.collection_name = chroma_config.get("collection_name")

        # Initialize optional LLM for RAG generation
        self.llm = RAGHelper.create_llm(self.config)

        # Configure search parameters (rrf_k unused but kept for API consistency)
        self.search_config = self.config.get("search", {})
        self.rrf_k = self.search_config.get("rrf_k", 60)

        logger.info("Initialized Chroma cost-optimized RAG search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute cost-optimized RAG search using dense semantic search.

        Performs semantic search using dense embeddings. Due to Chroma's
        limitations, this pipeline does not implement hybrid fusion.

        Args:
            query: Search query text.
            top_k: Number of results to return (default: 10).
            filters: Optional metadata filters as dictionary.
                Example: {"category": "tech", "published": True}

        Returns:
            Dictionary containing:
            - documents: List of retrieved documents with metadata
            - query: Original search query
            - answer: Generated RAG answer (if LLM configured)

        Raises:
            RuntimeError: If search fails due to API or database errors.

        Example:
            >>> pipeline = ChromaCostOptimizedRAGSearchPipeline(config)
            >>> result = pipeline.search("neural networks", top_k=5)
            >>> print(f"Found {len(result['documents'])} documents")
        """
        # Generate dense embedding for semantic search
        dense_query_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        logger.info("Embedded query with dense embedding: %s", query[:50])

        self.db._get_collection(self.collection_name)
        results_dict = self.db.query(
            query_embedding=dense_query_embedding,
            n_results=top_k,
            where=filters,
        )
        dense_documents = (
            ChromaDocumentConverter.convert_query_results_to_langchain_documents(
                results_dict
            )
        )
        logger.info("Retrieved %d documents from dense search", len(dense_documents))

        # Prepare result with retrieved documents
        result = {
            "documents": dense_documents[:top_k],
            "query": query,
        }

        # Generate RAG answer if LLM is configured
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, dense_documents[:top_k])
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
