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

from typing import Any

from langchain_core.documents import Document

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class ChromaQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
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

    @property
    def _db_key(self) -> str:
        return "chroma"

    def _initialize_db(self, db_config: dict[str, Any]) -> ChromaVectorDB:
        self.collection_name = db_config.get("collection_name")
        return ChromaVectorDB(persist_dir=db_config.get("persist_dir"))

    def _perform_search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[Document]:
        return self.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
        )
