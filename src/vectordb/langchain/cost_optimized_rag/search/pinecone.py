"""Pinecone cost-optimized RAG search pipeline for LangChain.

This module implements a cost-optimized search pipeline for Pinecone that
combines dense semantic search with sparse lexical search using RRF fusion.

Cost Optimization:
    - Single dense query embedding: Only one API call for semantic search
    - Local sparse embedding: TF-IDF generated locally (zero cost)
    - RRF fusion: Lightweight local algorithm, no API calls
    - Optional LLM: RAG generation can be disabled for retrieval-only

Search Strategy:
    1. Generate dense and sparse query embeddings
    2. Execute parallel dense and sparse searches in Pinecone
    3. Merge results using Reciprocal Rank Fusion (RRF)
    4. Optionally generate RAG answer with LLM

Example:
    >>> pipeline = PineconeCostOptimizedRAGSearchPipeline("config.yaml")
    >>> result = pipeline.search(
    ...     "What is machine learning?", top_k=10, filters={"category": "tech"}
    ... )
    >>> for doc in result["documents"]:
    ...     print(doc["metadata"]["text"][:200])
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    HaystackToLangchainConverter,
    RAGHelper,
    ResultMerger,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class PineconeCostOptimizedRAGSearchPipeline:
    """Pinecone search pipeline for cost-optimized RAG using LangChain.

    This pipeline implements hybrid search for Pinecone vector database,
    combining dense semantic search with sparse lexical search. It uses
    Reciprocal Rank Fusion (RRF) to merge results without requiring
    additional API calls.

    Attributes:
        config: Pipeline configuration dictionary.
        dense_embedder: API-based dense embedding model.
        sparse_embedder: Local sparse embedding generator.
        db: PineconeVectorDB client instance.
        index_name: Name of the Pinecone index.
        namespace: Namespace for document organization.
        llm: Optional language model for RAG generation.
        search_config: Search-specific configuration.
        rrf_k: RRF fusion parameter controlling rank weighting.

    Cost Benefits:
        - One dense embedding API call per query
        - Zero-cost sparse embedding (local)
        - Zero-cost fusion algorithm (local RRF)
        - Optional LLM calls only when answer generation needed

    Example:
        >>> pipeline = PineconeCostOptimizedRAGSearchPipeline("config.yaml")
        >>> result = pipeline.search("latest AI developments", top_k=5)
        >>> if "answer" in result:
        ...     print(result["answer"])
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Pinecone search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain pinecone section with api_key and index_name.
                Must contain embedding section with provider and model.
                Optionally contains llm section for RAG generation.
                Optionally contains search section with rrf_k parameter.

        Raises:
            ValueError: If required configuration is missing or invalid.
            FileNotFoundError: If config_or_path is a file path that does not exist.

        Note:
            LLM is optional. If not configured, search returns documents only
            without generated answers, reducing costs further.
        """
        # Load and validate configuration
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        # Initialize embedders for query processing
        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        # Configure Pinecone connection
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        # Initialize optional LLM for RAG generation
        self.llm = RAGHelper.create_llm(self.config)

        # Configure search parameters
        self.search_config = self.config.get("search", {})
        self.rrf_k = self.search_config.get("rrf_k", 60)

        logger.info(
            "Initialized Pinecone cost-optimized RAG search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute cost-optimized RAG search with hybrid fusion.

        Performs hybrid search by combining dense semantic and sparse lexical
        results using Reciprocal Rank Fusion (RRF). Optionally generates
        a RAG answer if an LLM is configured.

        Args:
            query: Search query text. Can be natural language or keyword-based.
            top_k: Number of results to return (default: 10).
            filters: Optional metadata filters as dictionary.
                Example: {"category": "tech", "date": {"$gte": "2024-01-01"}}

        Returns:
            Dictionary containing search results:
            - documents: List of retrieved documents with metadata
            - query: The original search query
            - answer: Generated RAG answer (if LLM configured)

        Raises:
            RuntimeError: If search fails due to API or database errors.

        Example:
            >>> pipeline = PineconeCostOptimizedRAGSearchPipeline(config)
            >>> result = pipeline.search(
            ...     "neural networks", top_k=5, filters={"topic": "AI"}
            ... )
            >>> print(f"Found {len(result['documents'])} documents")
        """
        # Generate dual embeddings for hybrid search
        # Dense via API for semantic understanding
        dense_query_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        # Sparse locally for lexical matching (zero cost)
        sparse_query_embedding = self.sparse_embedder.embed_query(query)
        logger.info(
            "Embedded query with both dense and sparse embeddings: %s", query[:50]
        )

        # Execute dense semantic search
        dense_documents = self.db.query(
            vector=dense_query_embedding,
            top_k=top_k,
            filter=filters,
            namespace=self.namespace,
        )
        dense_documents = HaystackToLangchainConverter.convert(dense_documents)
        logger.info("Retrieved %d documents from dense search", len(dense_documents))

        # Execute sparse lexical search
        # Note: Pinecone requires a placeholder dense vector for sparse-only queries
        sparse_documents = self.db.query_with_sparse(
            vector=[0.0]
            * self.dimension,  # Placeholder dense vector (not used in sparse search)
            sparse_vector=sparse_query_embedding,
            top_k=top_k,
            filter=filters,
            namespace=self.namespace,
        )
        sparse_documents = HaystackToLangchainConverter.convert(sparse_documents)
        logger.info("Retrieved %d documents from sparse search", len(sparse_documents))

        # Fuse results using Reciprocal Rank Fusion (RRF)
        # This merges dense and sparse results without additional API calls
        merged_documents = ResultMerger.merge_and_deduplicate(
            [dense_documents, sparse_documents],
            method="rrf",
            weights=[0.5, 0.5],
        )
        logger.info(
            "Fused %d unique documents from both searches", len(merged_documents)
        )

        # Prepare result with top-k documents
        result = {
            "documents": merged_documents[:top_k],
            "query": query,
        }

        # Generate RAG answer if LLM is configured (optional cost)
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, merged_documents[:top_k])
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
