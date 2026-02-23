"""Evaluation utilities for cost-optimized RAG pipelines.

Metrics computation and quality assessment for RAG retrieval systems.
Provides standardized evaluation across different vector databases
with cost-aware benchmarking capabilities.

Cost Optimization in Evaluation:

    Sampling Strategy:
        - Evaluate on representative query subset
        - Full evaluation on critical queries only
        - Progressive evaluation (coarse → fine)

    Metric Efficiency:
        - MRR, NDCG: Single pass computation
        - Recall@K, Precision@K: O(n) per query
        - Batch evaluation amortizes setup costs

    Benchmark Cost Control:
        - Limited query counts for latency tests
        - Synthetic data for throughput benchmarks
        - Cost estimation without full execution

Evaluation Strategy:

    Development Phase:
        - Quick smoke tests (10-50 queries)
        - Focus on Recall@10 and MRR
        - Local evaluation only

    Production Validation:
        - Full test suite (1000+ queries)
        - All metrics (MRR, NDCG, Recall, Precision)
        - Cost benchmarking included

    Regression Testing:
        - Subset of queries for CI/CD
        - Key metric thresholds
        - Automated pass/fail

When to Evaluate:
    - After index changes (validate quality)
    - Before deployment (performance verification)
    - During optimization (cost/quality trade-offs)
    - Periodically (detect drift)

Metrics Reference:
    - MRR: Mean Reciprocal Rank (0-1, higher better)
    - NDCG: Normalized Discounted Cumulative Gain
    - Recall@K: Proportion of relevant docs retrieved
    - Precision@K: Proportion of retrieved docs relevant
"""

from vectordb.haystack.cost_optimized_rag.evaluation.evaluator import (
    RAGEvaluator,
)


__all__ = ["RAGEvaluator"]
