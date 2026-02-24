"""MMR (Maximal Marginal Relevance) pipelines with indexing and search capabilities.

This module provides complete MMR pipeline implementations for diversity-aware
retrieval across all supported vector databases. MMR balances relevance with
diversity to reduce redundancy in search results.

MMR Algorithm Overview:
    MMR selects documents by scoring each candidate with:

        MMR_score = λ × relevance(query, doc) - (1-λ) × max_similarity(doc, selected)

    Where λ (lambda_param) controls the trade-off:
    - λ close to 1.0: Prioritize relevance (standard retrieval)
    - λ close to 0.5: Balanced relevance and diversity
    - λ close to 0.0: Prioritize diversity

    The algorithm greedily selects documents with highest MMR scores,
    penalizing documents similar to already-selected ones.

Lambda Parameter Tuning:
    - λ = 0.7-0.8: Emphasize relevance, good for precise queries
    - λ = 0.5: Balanced, good default for most use cases
    - λ = 0.3-0.4: Emphasize diversity, good for exploratory search

Pipelines Provided:
    - Indexing Pipelines: Load, embed, and index documents for MMR retrieval
    - Search Pipelines: Embed query, retrieve candidates, apply MMR reranking

Example:
    >>> from vectordb.haystack.mmr import ChromaMmrSearchPipeline
    >>> pipeline = ChromaMmrSearchPipeline("mmr_config.yaml")
    >>> results = pipeline.search("quantum computing applications", top_k=10)
    >>> print(
    ...     f"Retrieved {results['candidates_retrieved']} candidates, "
    ...     f"returned {results['documents_after_mmr']} diverse results"
    ... )
"""

from vectordb.haystack.mmr.indexing import (
    ChromaMmrIndexingPipeline,
    MilvusMmrIndexingPipeline,
    PineconeMmrIndexingPipeline,
    QdrantMmrIndexingPipeline,
    WeaviateMmrIndexingPipeline,
)
from vectordb.haystack.mmr.search import (
    ChromaMmrSearchPipeline,
    MilvusMmrSearchPipeline,
    PineconeMmrSearchPipeline,
    QdrantMmrSearchPipeline,
    WeaviateMmrSearchPipeline,
)


__all__ = [
    # Indexing
    "ChromaMmrIndexingPipeline",
    "MilvusMmrIndexingPipeline",
    "PineconeMmrIndexingPipeline",
    "QdrantMmrIndexingPipeline",
    "WeaviateMmrIndexingPipeline",
    # Search
    "ChromaMmrSearchPipeline",
    "MilvusMmrSearchPipeline",
    "PineconeMmrSearchPipeline",
    "QdrantMmrSearchPipeline",
    "WeaviateMmrSearchPipeline",
]
