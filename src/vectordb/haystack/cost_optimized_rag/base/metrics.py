"""Retrieval evaluation metrics for cost-quality trade-off analysis.

Provides standard IR metrics to validate that cost optimizations don't
unacceptably degrade retrieval quality. All metrics support the cost-
optimization goal by enabling evidence-based tuning of hyperparameters.

Key Metrics for Cost Optimization:
    - Recall@K: Measures if relevant documents are in the candidate set.
      Lower top_k values (for cost savings) must maintain adequate recall.
    - NDCG@K: Ranks quality of top-K results. Validates that aggressive
      reranking doesn't harm result ordering significantly.
    - MRR: Mean Reciprocal Rank for single-relevant-document scenarios.
      Critical for question-answering use cases.

Usage Pattern:
    Use MetricsAggregator to collect per-query metrics, then compute
    averages to validate configuration changes. If reducing top_k from
    20 to 10 drops recall@10 by <5%, the cost savings are justified.

Trade-off Analysis:
    - Embedding model size vs Recall: Track recall when switching from
      large (768-dim) to small (384-dim) models
    - Quantization vs NDCG: Measure ranking quality degradation from
      scalar/binary quantization for storage cost savings
    - Sparse-first vs Dense-only: Compare hybrid search (lower cost)
      against pure dense retrieval (higher cost, potentially better recall)
"""

import math
from typing import Any


class RetrievalMetrics:
    """Calculate retrieval quality metrics."""

    @staticmethod
    def mrr(
        retrieved_ids: list[str],
        relevant_ids: set[str],
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / rank of first relevant result.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.

        Returns:
            MRR score (0-1). 0 if no relevant document found.
        """
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        """Calculate Recall@K.

        Recall@K = |retrieved ∩ relevant| / |relevant| (top-K only).

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.
            k: Cutoff for top-K.

        Returns:
            Recall@K score (0-1).
        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        return len(set(top_k) & relevant_ids) / len(relevant_ids)

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        """Calculate Precision@K.

        Precision@K = |retrieved ∩ relevant| / k (top-K only).

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.
            k: Cutoff for top-K.

        Returns:
            Precision@K score (0-1).
        """
        if k <= 0:
            return 0.0
        top_k = retrieved_ids[:k]
        return len(set(top_k) & relevant_ids) / k

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        NDCG@K = DCG@K / IDCG@K, where:
        - DCG = Σ(rel_i / log2(i+1)) for i in top-K
        - IDCG = perfect ranking DCG

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.
            k: Cutoff for top-K.

        Returns:
            NDCG@K score (0-1).
        """
        top_k = retrieved_ids[:k]

        # DCG: Σ (rel_i / log2(i+1))
        dcg = 0.0
        for i, doc_id in enumerate(top_k, 1):
            rel = 1.0 if doc_id in relevant_ids else 0.0
            dcg += rel / math.log2(i + 1)

        # IDCG: perfect ranking
        ideal_k = min(k, len(relevant_ids))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def map_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> float:
        """Calculate Mean Average Precision (MAP@K).

        MAP@K = Σ(precision@i * rel_i) / min(k, |relevant|)

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.
            k: Cutoff for top-K.

        Returns:
            MAP@K score (0-1).
        """
        if not relevant_ids:
            return 0.0

        top_k = retrieved_ids[:k]
        precision_sum = 0.0
        num_relevant = 0

        for i, doc_id in enumerate(top_k, 1):
            if doc_id in relevant_ids:
                precision_sum += RetrievalMetrics.precision_at_k(
                    retrieved_ids, relevant_ids, i
                )
                num_relevant += 1

        return precision_sum / min(k, len(relevant_ids))

    @staticmethod
    def compute_all(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> dict[str, float]:
        """Compute all metrics at once.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.
            k: Cutoff for top-K metrics.

        Returns:
            Dictionary with all metric scores.
        """
        return {
            "mrr": RetrievalMetrics.mrr(retrieved_ids, relevant_ids),
            "recall_at_k": RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k),
            "precision_at_k": RetrievalMetrics.precision_at_k(
                retrieved_ids, relevant_ids, k
            ),
            "ndcg_at_k": RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k),
            "map_at_k": RetrievalMetrics.map_at_k(retrieved_ids, relevant_ids, k),
        }


class MetricsAggregator:
    """Aggregate metrics across multiple queries."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self.results: list[dict[str, Any]] = []

    def add_result(
        self,
        query_id: str,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int = 10,
    ) -> None:
        """Add a retrieval result for aggregation.

        Args:
            query_id: Unique query identifier.
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids: Set of relevant document IDs.
            k: Cutoff for top-K metrics.
        """
        metrics = RetrievalMetrics.compute_all(retrieved_ids, relevant_ids, k)
        self.results.append(
            {
                "query_id": query_id,
                "retrieved_count": len(retrieved_ids),
                "relevant_count": len(relevant_ids),
                **metrics,
            }
        )

    def aggregate(self) -> dict[str, float]:
        """Compute aggregate statistics.

        Returns:
            Dictionary with averaged metrics.

        Raises:
            ValueError: If no results aggregated.
        """
        if not self.results:
            msg = "No results to aggregate"
            raise ValueError(msg)

        metrics_keys = [
            "mrr",
            "recall_at_k",
            "precision_at_k",
            "ndcg_at_k",
            "map_at_k",
        ]

        aggregates = {}
        for metric_key in metrics_keys:
            values = [r[metric_key] for r in self.results]
            aggregates[f"avg_{metric_key}"] = sum(values) / len(values)

        return aggregates

    def summary(self) -> dict[str, Any]:
        """Get aggregation summary.

        Returns:
            Dictionary with aggregated metrics and counts.
        """
        if not self.results:
            return {"total_queries": 0}

        agg = self.aggregate()
        return {
            "total_queries": len(self.results),
            **agg,
        }
