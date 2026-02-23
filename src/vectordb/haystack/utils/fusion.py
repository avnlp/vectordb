"""Result fusion strategies for Haystack hybrid search pipelines.

This module implements fusion algorithms for combining results from multiple
retrieval sources in Haystack pipelines. Fusion is essential for hybrid search
where dense (semantic) and sparse (BM25/keyword) retrievers are used together.

Fusion Strategies:
    Reciprocal Rank Fusion (RRF):
        Combines rankings using the formula: score = 1/(k + rank)
        RRF is robust because it uses ranks rather than raw scores, making it
        effective when combining retrievers with different score distributions.
        The k parameter (default 60) controls how much weight is given to lower-
        ranked documents.

    Weighted Fusion:
        Combines inverse-rank scores with configurable weights for each source.
        Use when you have prior knowledge about retriever quality, e.g., giving
        more weight to dense retrieval for semantic queries.

When to Use Each:
    - RRF (fuse_rrf): Default choice for hybrid search. No tuning required.
    - Weighted (fuse_weighted): When one retriever is known to be more reliable.

Document Deduplication:
    Both fusion methods deduplicate results using document IDs or content
    previews, ensuring each document appears only once in the final ranking.

Usage:
    >>> from vectordb.haystack.utils import ResultMerger
    >>> fused = ResultMerger.fuse_rrf(dense_results, sparse_results, top_k=10)
    >>> # Or with weights
    >>> fused = ResultMerger.fuse_weighted(
    ...     dense_results, sparse_results, dense_weight=0.7, sparse_weight=0.3
    ... )
"""

import logging
from typing import Any

from haystack import Document


logger = logging.getLogger(__name__)


class ResultMerger:
    """Utility class for merging hybrid search results.

    Provides class methods for fusing dense and sparse retrieval results
    using RRF or weighted strategies. All methods handle deduplication
    automatically.
    """

    @classmethod
    def fuse_rrf(
        cls,
        dense_results: list[Document],
        sparse_results: list[Document],
        top_k: int = 10,
        k: float = 60.0,
    ) -> list[Document]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank)) for each ranker.

        Args:
            dense_results: Documents from dense retrieval (ordered by relevance).
            sparse_results: Documents from sparse/BM25 retrieval (ordered by relevance).
            top_k: Number of results to return.
            k: RRF parameter (default 60). Lower k emphasizes position more.

        Returns:
            List of fused and re-ranked documents (up to top_k).
        """
        fused_scores: dict[str, tuple[float, Document]] = {}

        # Score dense results
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc.id or doc.content[:50]  # Fallback to content preview
            score = 1.0 / (k + rank)
            if doc_id in fused_scores:
                prev_score, _ = fused_scores[doc_id]
                fused_scores[doc_id] = (prev_score + score, doc)
            else:
                fused_scores[doc_id] = (score, doc)

        # Score sparse results
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc.id or doc.content[:50]
            score = 1.0 / (k + rank)
            if doc_id in fused_scores:
                prev_score, prev_doc = fused_scores[doc_id]
                fused_scores[doc_id] = (prev_score + score, prev_doc)
            else:
                fused_scores[doc_id] = (score, doc)

        fused = sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)
        results = [doc for _, (_, doc) in fused[:top_k]]

        logger.debug(
            "Fused %d dense + %d sparse results into %d",
            len(dense_results),
            len(sparse_results),
            len(results),
        )
        return results

    @classmethod
    def fuse_weighted(
        cls,
        dense_results: list[Document],
        sparse_results: list[Document],
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[Document]:
        """Fuse dense and sparse results with weighted scoring.

        Args:
            dense_results: Documents from dense retrieval.
            sparse_results: Documents from sparse/BM25 retrieval.
            top_k: Number of results to return.
            dense_weight: Weight for dense retrieval (0.0-1.0).
            sparse_weight: Weight for sparse retrieval (0.0-1.0).

        Returns:
            List of fused and re-ranked documents (up to top_k).
        """
        fused_scores: dict[str, tuple[float, Document]] = {}

        # Normalize weights
        total_weight = dense_weight + sparse_weight
        dense_weight /= total_weight
        sparse_weight /= total_weight

        # Score dense results (inverse rank)
        for rank, doc in enumerate(dense_results, 1):
            doc_id = doc.id or doc.content[:50]
            score = (1.0 / rank) * dense_weight
            if doc_id in fused_scores:
                prev_score, _ = fused_scores[doc_id]
                fused_scores[doc_id] = (prev_score + score, doc)
            else:
                fused_scores[doc_id] = (score, doc)

        # Score sparse results
        for rank, doc in enumerate(sparse_results, 1):
            doc_id = doc.id or doc.content[:50]
            score = (1.0 / rank) * sparse_weight
            if doc_id in fused_scores:
                prev_score, prev_doc = fused_scores[doc_id]
                fused_scores[doc_id] = (prev_score + score, prev_doc)
            else:
                fused_scores[doc_id] = (score, doc)

        fused = sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)
        results = [doc for _, (_, doc) in fused[:top_k]]

        logger.debug(
            "Fused %d dense + %d sparse results (weights: %.2f/%.2f) into %d",
            len(dense_results),
            len(sparse_results),
            dense_weight,
            sparse_weight,
            len(results),
        )
        return results

    @classmethod
    def fuse(
        cls,
        dense_results: list[Document],
        sparse_results: list[Document],
        top_k: int = 10,
        strategy: str = "rrf",
        **kwargs: Any,
    ) -> list[Document]:
        """Fuse results using specified strategy.

        Args:
            dense_results: Documents from dense retrieval.
            sparse_results: Documents from sparse/BM25 retrieval.
            top_k: Number of results to return.
            strategy: Fusion strategy ('rrf' or 'weighted').
            **kwargs: Additional parameters for the fusion strategy.

        Returns:
            List of fused and re-ranked documents (up to top_k).

        Raises:
            ValueError: If unknown strategy specified.
        """
        if strategy == "rrf":
            k = kwargs.get("k", 60.0)
            return cls.fuse_rrf(dense_results, sparse_results, top_k=top_k, k=k)
        if strategy == "weighted":
            dense_weight = kwargs.get("dense_weight", 0.7)
            sparse_weight = kwargs.get("sparse_weight", 0.3)
            return cls.fuse_weighted(
                dense_results,
                sparse_results,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        msg = f"Unknown fusion strategy: {strategy}"
        raise ValueError(msg)
