"""Evaluation metrics for retrieval pipelines.

This module provides standard information retrieval metrics for measuring
the quality of document retrieval in RAG pipelines. All metrics are designed
to work with ground truth relevance judgments.

Metrics Provided:
    - Recall@k: Fraction of relevant documents retrieved in top-k
    - Precision@k: Fraction of top-k documents that are relevant
    - MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant doc
    - NDCG@k (Normalized DCG): Rank-aware metric normalized by ideal ranking
    - Hit Rate: Binary indicator if any relevant doc is in top-k

Metric Formulas:
    Recall@k = |relevant ∩ retrieved_top_k| / |relevant|
    Precision@k = |relevant ∩ retrieved_top_k| / k
    MRR = mean(1 / rank_of_first_relevant)
    NDCG@k = DCG@k / IDCG@k (where DCG = Σ rel_i / log2(i+2))
    Hit Rate = 1 if any relevant in top-k, else 0

Design Notes:
    - Metrics assume binary relevance (document is relevant or not)
    - All metrics use 1-indexed ranks for mathematical correctness
    - NDCG uses ideal DCG (IDCG) for normalization

Usage:
    >>> from vectordb.utils.evaluation import compute_mrr, compute_ndcg_at_k
    >>> mrr = compute_mrr(["doc1", "doc2"], {"doc1", "doc3"})
    >>> ndcg = compute_ndcg_at_k(["doc1", "doc2"], {"doc1"}, k=10)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics.

    Attributes:
        recall_at_k: Proportion of relevant documents retrieved in top-k.
        precision_at_k: Proportion of top-k documents that are relevant.
        mrr: Mean Reciprocal Rank - average of reciprocal of first relevant rank.
        ndcg_at_k: Normalized Discounted Cumulative Gain at k.
        hit_rate: Proportion of queries with at least one relevant doc in top-k.
        num_queries: Number of queries evaluated.
        k: The cutoff value for top-k metrics.
    """

    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    num_queries: int = 0
    k: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format.

        Returns:
            Dictionary with all metric values.
        """
        return {
            f"recall@{self.k}": self.recall_at_k,
            f"precision@{self.k}": self.precision_at_k,
            "mrr": self.mrr,
            f"ndcg@{self.k}": self.ndcg_at_k,
            "hit_rate": self.hit_rate,
            "num_queries": self.num_queries,
        }


@dataclass
class QueryResult:
    """Result for a single query evaluation.

    Attributes:
        query: The query string.
        retrieved_ids: List of retrieved document IDs.
        retrieved_contents: List of retrieved document contents.
        relevant_ids: Set of ground truth relevant document IDs.
        scores: List of retrieval scores for each document.
    """

    query: str
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_contents: list[str] = field(default_factory=list)
    relevant_ids: set[str] = field(default_factory=set)
    scores: list[float] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a retrieval pipeline.

    Attributes:
        metrics: Aggregated retrieval metrics.
        query_results: Per-query detailed results.
        pipeline_name: Name of the evaluated pipeline.
        dataset_name: Name of the evaluation dataset.
        config: Configuration used for the evaluation.
    """

    metrics: RetrievalMetrics
    query_results: list[QueryResult] = field(default_factory=list)
    pipeline_name: str = ""
    dataset_name: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation result to dictionary format.

        Returns:
            Dictionary with metrics and metadata.
        """
        return {
            "pipeline": self.pipeline_name,
            "dataset": self.dataset_name,
            "metrics": self.metrics.to_dict(),
            "num_queries": len(self.query_results),
        }


def compute_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Recall@k for a single query.

    Recall@k = (relevant docs in top-k) / (total relevant docs)

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall score between 0 and 1.
    """
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    relevant_retrieved = top_k.intersection(relevant_ids)
    return len(relevant_retrieved) / len(relevant_ids)


def compute_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Precision@k for a single query.

    Precision@k = (relevant docs in top-k) / k

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision score between 0 and 1.
    """
    if k == 0:
        return 0.0

    top_k = set(retrieved_ids[:k])
    relevant_retrieved = top_k.intersection(relevant_ids)
    return len(relevant_retrieved) / k


def compute_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Compute Mean Reciprocal Rank for a single query.

    MRR = 1 / (rank of first relevant document)

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of ground truth relevant document IDs.

    Returns:
        Reciprocal rank score between 0 and 1.
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_dcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain at k.

    DCG@k = sum(rel_i / log2(i + 2)) for i in range(k)

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        DCG score.
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # +2 because log2(1) = 0
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k (Ideal DCG)

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        NDCG score between 0 and 1.
    """
    dcg = compute_dcg_at_k(retrieved_ids, relevant_ids, k)

    # Ideal DCG: all relevant docs ranked first
    ideal_retrieved = list(relevant_ids)[:k]
    idcg = compute_dcg_at_k(ideal_retrieved, relevant_ids, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_hit_rate(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute hit rate (binary success) for a single query.

    Hit rate = 1 if any relevant doc in top-k, else 0

    Args:
        retrieved_ids: List of retrieved document IDs in ranked order.
        relevant_ids: Set of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k.intersection(relevant_ids) else 0.0


def evaluate_retrieval(
    query_results: list[QueryResult],
    k: int = 5,
) -> RetrievalMetrics:
    """Evaluate retrieval quality over multiple queries.

    Computes aggregated metrics over all provided query results.

    Args:
        query_results: List of QueryResult objects with retrieved and relevant IDs.
        k: Cutoff for top-k metrics.

    Returns:
        RetrievalMetrics with averaged scores.
    """
    if not query_results:
        return RetrievalMetrics(k=k)

    recalls = []
    precisions = []
    mrrs = []
    ndcgs = []
    hits = []

    for result in query_results:
        recalls.append(
            compute_recall_at_k(result.retrieved_ids, result.relevant_ids, k)
        )
        precisions.append(
            compute_precision_at_k(result.retrieved_ids, result.relevant_ids, k)
        )
        mrrs.append(compute_mrr(result.retrieved_ids, result.relevant_ids))
        ndcgs.append(compute_ndcg_at_k(result.retrieved_ids, result.relevant_ids, k))
        hits.append(compute_hit_rate(result.retrieved_ids, result.relevant_ids, k))

    return RetrievalMetrics(
        recall_at_k=float(np.mean(recalls)),
        precision_at_k=float(np.mean(precisions)),
        mrr=float(np.mean(mrrs)),
        ndcg_at_k=float(np.mean(ndcgs)),
        hit_rate=float(np.mean(hits)),
        num_queries=len(query_results),
        k=k,
    )
