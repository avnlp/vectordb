"""MMR search pipelines for all vector databases.

This module provides Haystack pipeline implementations for MMR-based retrieval.
Each search pipeline follows a consistent pattern:

1. Embed the query using the configured text embedder
2. Retrieve candidates from the vector database (higher top_k for more candidates)
3. Apply MMR reranking to balance relevance and diversity
4. Optionally generate RAG answer if enabled

MMR Algorithm:
    Maximal Marginal Relevance balances query relevance with result diversity:

        MMR_score = λ × similarity(query, doc) - (1-λ) × max_similarity(doc, selected)

    Where λ (lambda_threshold) controls the trade-off:
    - λ = 1.0: Pure relevance ranking (standard retrieval)
    - λ = 0.5: Balanced relevance and diversity (default)
    - λ = 0.0: Pure diversity ranking

    The algorithm greedily selects documents with highest MMR scores,
    penalizing documents similar to already-selected ones.

    MMR Steps:
    1. Calculate query-document similarity for all candidates
    2. For first document: select highest query similarity
    3. For subsequent documents:
       a. Calculate max similarity to already-selected documents
       b. Compute MMR score: λ×relevance - (1-λ)×redundancy
       c. Select document with highest MMR score
    4. Repeat until top_k documents selected

Key Parameter:
    top_k_candidates (usually 2-3x the final top_k) determines how many
    documents are retrieved before MMR reranking. Higher values give MMR
    more options to select a diverse set, but increase latency.

Supported Databases:
    All MMR search pipelines support the same configuration structure with
    database-specific connection parameters:
    - ChromaMmrSearchPipeline: Local embedded database
    - PineconeMmrSearchPipeline: Managed cloud with namespaces
    - MilvusMmrSearchPipeline: Scalable open-source
    - QdrantMmrSearchPipeline: High-performance with rich filtering
    - WeaviateMmrSearchPipeline: Graph-based with semantic capabilities

Example:
    >>> from vectordb.haystack.mmr.search import ChromaMmrSearchPipeline
    >>> pipeline = ChromaMmrSearchPipeline("mmr_config.yaml")
    >>> results = pipeline.search("machine learning applications", top_k=10)
    >>> print(
    ...     f"Retrieved {results['candidates_retrieved']}, "
    ...     f"returned {results['documents_after_mmr']} diverse results"
    ... )

Note:
    All search pipelines return metadata including candidates_retrieved
    and documents_after_mmr for debugging and evaluation.
"""

from vectordb.haystack.mmr.search.chroma import ChromaMmrSearchPipeline
from vectordb.haystack.mmr.search.milvus import MilvusMmrSearchPipeline
from vectordb.haystack.mmr.search.pinecone import PineconeMmrSearchPipeline
from vectordb.haystack.mmr.search.qdrant import QdrantMmrSearchPipeline
from vectordb.haystack.mmr.search.weaviate import WeaviateMmrSearchPipeline


__all__ = [
    "ChromaMmrSearchPipeline",
    "MilvusMmrSearchPipeline",
    "PineconeMmrSearchPipeline",
    "QdrantMmrSearchPipeline",
    "WeaviateMmrSearchPipeline",
]
