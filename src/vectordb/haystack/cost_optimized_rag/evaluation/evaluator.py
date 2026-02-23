"""Evaluation pipeline for RAG retrieval metrics.

Computes standard IR metrics (MRR, NDCG, Recall, Precision) for RAG
retrieval quality assessment. Includes benchmarking tools for
performance and cost estimation.

Cost-Aware Evaluation Strategies:

    Progressive Evaluation:
        1. Smoke test: 10 queries, basic metrics only
        2. Validation: 100 queries, all metrics
        3. Full test: 1000+ queries, benchmarks included
        - Costs scale linearly with query count
        - Early termination on critical failures

    Metric Selection:
        - MRR: Single best result ranking (cheap)
        - Recall@K: Coverage metric (moderate)
        - NDCG: Position-aware quality (expensive)
        - Choose based on evaluation goals

    Benchmark Sampling:
        - Throughput: Synthetic queries (no LLM cost)
        - Latency: Real queries (embedding cost only)
        - Cost: Estimation without full execution

Metrics Computation:

    MRR (Mean Reciprocal Rank):
        Formula: average(1 / rank_of_first_relevant)
        Best case: 1.0 (first result always relevant)
        Good: >0.5, Acceptable: >0.3

    NDCG (Normalized DCG):
        Measures ranking quality with position discount
        Range: 0-1 (1 is perfect ranking)
        Good: >0.6, Acceptable: >0.4

    Recall@K:
        Formula: |relevant ∩ retrieved| / |relevant|
        Measures coverage of relevant documents
        Good: >0.7, Acceptable: >0.5

    Precision@K:
        Formula: |relevant ∩ retrieved| / |retrieved|
        Measures relevance of retrieved documents
        Good: >0.5, Acceptable: >0.3

Cost Estimation Methodology:

    Rough cost model based on typical pricing:
        - Database: Self-hosted ($0) vs Cloud ($/GB)
        - Embeddings: OpenAI ($0.02/1M tokens) vs Local ($0)
        - Queries: Varies by model and usage

    Estimation only - update with actual pricing for accuracy.
"""

import logging
import time
from typing import Any

from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig
from vectordb.haystack.cost_optimized_rag.base.metrics import (
    MetricsAggregator,
    RetrievalMetrics,
)
from vectordb.utils.logging import LoggerFactory


class RAGEvaluator:
    """Evaluate RAG retrieval quality and performance.

    Computes standard IR metrics with latency tracking.
    Designed for cost-efficient evaluation through
    configurable sample sizes and metric selection.

    Cost Architecture:
        - Evaluation cost = queries × (search_cost + metric_compute)
        - Metrics: O(n) per query (n = retrieved docs)
        - Latency tracking: Minimal overhead

    Performance Characteristics:
        - Single query: ~100-500ms (depends on searcher)
        - Batch evaluation: Parallelizable
        - Memory: O(batch_size × top_k)

    Usage Pattern:
        >>> evaluator = RAGEvaluator(config, searcher)
        >>> result = evaluator.evaluate_query(
        ...     query_id="q1",
        ...     query="test query",
        ...     relevant_ids={"doc1", "doc2"},
        ...     top_k=10,
        ... )
        >>> print(result["mrr"], result["recall_at_k"])

    Example:
        >>> queries = [
        ...     {"query_id": "q1", "query": "test", "relevant_ids": ["d1"]},
        ... ]
        >>> summary = evaluator.evaluate_batch(queries)
        >>> print(summary["mean_mrr"])
    """

    def __init__(self, config: RAGConfig, searcher: Any) -> None:
        """Initialize evaluator.

        Args:
            config: RAGConfig with evaluation settings.
            searcher: Initialized searcher instance (Pinecone, Qdrant, etc.).
        """
        self.config = config
        self.searcher = searcher
        self.logger = LoggerFactory(
            f"{config.logging.name}_evaluator",
            log_level=getattr(logging, config.logging.level.upper()),
        ).get_logger()
        self.aggregator = MetricsAggregator()

    def evaluate_query(
        self,
        query_id: str,
        query: str,
        relevant_ids: set[str],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate single query with metrics and latency.

        Executes search and computes all retrieval metrics.
        Tracks latency for performance monitoring.

        Args:
            query_id: Query identifier for tracking.
            query: Query text to evaluate.
            relevant_ids: Set of relevant document IDs.
            top_k: Number of results to evaluate.

        Returns:
            Dict with query metrics (mrr, ndcg, recall, precision, latency).
        """
        if top_k is None:
            top_k = self.config.search.top_k

        start_time = time.time()
        results = self.searcher.search(query, top_k=top_k)
        latency = time.time() - start_time

        retrieved_ids = [r.get("id", "") for r in results]
        metrics = RetrievalMetrics.compute_all(retrieved_ids, relevant_ids, top_k)

        self.aggregator.add_result(query_id, retrieved_ids, relevant_ids, top_k)

        return {
            "query_id": query_id,
            "query": query,
            "latency_ms": latency * 1000,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_ids),
            **metrics,
        }

    def evaluate_batch(
        self,
        queries: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate batch of queries with aggregation.

        Processes queries sequentially and computes summary statistics.
        Cost scales linearly with query count.

        Args:
            queries: List of dicts with query_id, query, relevant_ids.
            top_k: Number of results to evaluate.

        Returns:
            Summary metrics with individual results.
        """
        results = []
        for q in queries:
            result = self.evaluate_query(
                q.get("query_id", ""),
                q.get("query", ""),
                set(q.get("relevant_ids", [])),
                top_k=top_k,
            )
            results.append(result)

        summary = self.aggregator.summary()
        summary["individual_results"] = results
        return summary

    def get_report(self) -> dict[str, Any]:
        """Get final evaluation report.

        Returns:
            Aggregated metrics and summary statistics.
        """
        return self.aggregator.summary()
