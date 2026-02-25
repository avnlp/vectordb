"""Chroma query enhancement search pipeline (LangChain).

This module implements the search pipeline for query enhancement using Chroma
as the vector database backend. Query enhancement improves retrieval quality
by generating multiple query variations before executing searches.

Query Enhancement Strategies:
    The pipeline supports three enhancement modes:

    Multi-Query Generation:
        - Generates 5 alternative phrasings of the original query
        - Addresses vocabulary mismatch by casting a wider semantic net
        - Each variation may match different documents in the corpus
        - Best for queries with domain-specific terminology

    HyDE (Hypothetical Document Embeddings):
        - Generates a hypothetical answer document
        - Uses the hypothetical document for retrieval instead of query
        - Bridges the gap between query and document distributions
        - Best for short queries or questions vs. documents mismatch

    Step-Back Prompting:
        - Generates 3 broader context questions
        - Retrieves background information before specific query
        - Best for complex questions requiring foundational knowledge

Pipeline Architecture:
    1. Query Enhancement: Use QueryEnhancer to generate multiple query variations
    2. Parallel Search: Execute similarity search for each variation
    3. Result Fusion: Merge results using Reciprocal Rank Fusion (RRF)
    4. Optional RAG: Generate answer using retrieved documents

Reciprocal Rank Fusion:
    RRF combines results from multiple queries by computing:
        RRF_score = sum(1.0 / (k + rank))

    This approach favors documents that appear in multiple result sets,
    reducing the impact of any single query variation.

Configuration:
    Requires standard Chroma config plus query_enhancement settings:
        query_enhancement:
          mode: "multi_query"  # or "hyde", "step_back"
          llm:
            provider: "groq"
            model: "llama-3.3-70b-versatile"
            temperature: 0.3

Example:
    >>> pipeline = ChromaQueryEnhancementSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="What is backpropagation?",
    ...     top_k=5,
    ...     mode="step_back",
    ... )
    >>> print(f"Retrieved {len(results['documents'])} documents")
    >>> print(f"Generated queries: {results['enhanced_queries']}")

Performance Notes:
    - Query enhancement increases retrieval time proportionally to enhancement mode
    - Multi-query: 5x the searches (5 parallel queries)
    - HyDE: 2x the searches + 1 LLM call
    - Step-back: 4x the searches
    - Parallel execution minimizes latency impact

See Also:
    vectordb.langchain.components.query_enhancer: Core enhancement component
    vectordb.langchain.query_enhancement.search: Search pipelines for all databases
    vectordb.langchain.query_enhancement.indexing: Indexing pipelines
"""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.components import QueryEnhancer
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    ResultMerger,
)


logger = logging.getLogger(__name__)


class ChromaQueryEnhancementSearchPipeline:
    """Chroma search pipeline with query enhancement (LangChain).

    Implements diversity-aware document retrieval by generating multiple query
    variations and fusing results using Reciprocal Rank Fusion. This approach
    improves recall by addressing vocabulary mismatch and query-document
    distribution gaps.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for query vectorization.
        db: ChromaVectorDB instance for local vector storage.
        collection_name: Name of Chroma collection to search.
        query_enhancer: QueryEnhancer instance for generating variations.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> config = {
        ...     "chroma": {
        ...         "persist_dir": "./chroma_db",
        ...         "collection_name": "documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ...     "query_enhancement": {
        ...         "mode": "multi_query",
        ...         "llm": {"model": "llama-3.3-70b-versatile"},
        ...     },
        ... }
        >>> pipeline = ChromaQueryEnhancementSearchPipeline(config)
        >>> results = pipeline.search("neural networks", top_k=10)
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize query enhancement search pipeline from configuration.

        Sets up Chroma connection, embedding model, LLM for query generation,
        and the query enhancer component.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain chroma, embedder, and query_enhancement sections.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_dir=chroma_config.get("persist_dir"),
        )

        self.collection_name = chroma_config.get("collection_name")

        # Initialize LLM for query enhancement
        llm = RAGHelper.create_llm(self.config)
        if llm is None:
            from langchain_groq import ChatGroq

            llm = ChatGroq(model="llama-3.3-70b-versatile")

        self.query_enhancer = QueryEnhancer(llm)

        # Optional RAG for answer generation
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Chroma query enhancement search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        mode: str = "multi_query",
    ) -> dict[str, Any]:
        """Execute query enhancement search against Chroma collection.

        Generates multiple query variations, performs parallel searches,
        and fuses results using Reciprocal Rank Fusion.

        Args:
            query: Search query text to enhance and execute.
            top_k: Number of final results to return after fusion.
            filters: Optional metadata filters for Chroma query.
            mode: Query enhancement mode - one of "multi_query", "hyde",
                or "step_back". Default is "multi_query".

        Returns:
            Dictionary containing:
                - documents: List of Document objects from fused results
                - query: Original search query
                - enhanced_queries: List of generated query variations
                - answer: Optional RAG-generated answer (if LLM configured)

        Example:
            >>> results = pipeline.search(
            ...     query="What is deep learning?",
            ...     top_k=10,
            ...     mode="multi_query",
            ... )
            >>> print(f"Enhanced queries: {results['enhanced_queries']}")
            >>> print(f"Found {len(results['documents'])} documents")
        """
        logger.info("Starting query enhancement search (mode=%s)", mode)

        # Generate enhanced query variations
        enhanced_queries = self.query_enhancer.generate_queries(query, mode=mode)
        logger.info("Generated %d enhanced queries", len(enhanced_queries))

        # Perform parallel searches for each query variation
        all_results: list[list[Document]] = []
        for enhanced_query in enhanced_queries:
            query_embedding = EmbedderHelper.embed_query(self.embedder, enhanced_query)
            documents = self.db.query(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                collection_name=self.collection_name,
            )
            all_results.append(documents)
            logger.info(
                "Retrieved %d documents for query: %s",
                len(documents),
                enhanced_query[:50],
            )

        # Fuse results using Reciprocal Rank Fusion
        fused_documents = ResultMerger.reciprocal_rank_fusion(all_results, k=60)
        fused_documents = fused_documents[:top_k]
        logger.info("Fused results: %d documents", len(fused_documents))

        result = {
            "documents": fused_documents,
            "query": query,
            "enhanced_queries": enhanced_queries,
        }

        # Generate RAG answer if LLM is configured
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, fused_documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
