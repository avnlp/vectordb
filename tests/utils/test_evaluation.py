"""Tests for retrieval evaluation metrics.

This module tests the evaluation framework for measuring retrieval quality.
The metrics follow standard information retrieval conventions and support
configurable cutoff values (k).

Tested metrics:
    - Recall@k: Proportion of relevant documents retrieved in top-k
    - Precision@k: Proportion of top-k documents that are relevant
    - MRR (Mean Reciprocal Rank): Average of 1/rank for first relevant doc
    - NDCG@k: Normalized Discounted Cumulative Gain
    - Hit Rate: Binary indicator of any relevant doc in top-k

Tested dataclasses:
    RetrievalMetrics: Aggregated metrics across queries
    QueryResult: Per-query retrieval results with ground truth
    EvaluationResult: Complete evaluation with pipeline metadata

Test coverage includes:
    - Perfect, partial, and zero-hit scenarios
    - Edge cases: empty results, empty ground truth, k variations
    - Multi-query aggregation
"""

import math

import pytest

from vectordb.utils.evaluation import (
    EvaluationResult,
    QueryResult,
    RetrievalMetrics,
    compute_dcg_at_k,
    compute_hit_rate,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_retrieval,
)


class TestRetrievalMetrics:
    """Test suite for RetrievalMetrics dataclass.

    Tests cover initialization with defaults/custom values and dictionary
    serialization with k-specific metric key formatting.
    """

    def test_initialization_defaults(self) -> None:
        """Test RetrievalMetrics with default values."""
        metrics = RetrievalMetrics()
        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg_at_k == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.num_queries == 0
        assert metrics.k == 5

    def test_initialization_custom(self) -> None:
        """Test RetrievalMetrics with custom values."""
        metrics = RetrievalMetrics(
            recall_at_k=0.85,
            precision_at_k=0.90,
            mrr=0.75,
            ndcg_at_k=0.88,
            hit_rate=0.95,
            num_queries=100,
            k=10,
        )
        assert metrics.recall_at_k == 0.85
        assert metrics.precision_at_k == 0.90
        assert metrics.mrr == 0.75
        assert metrics.ndcg_at_k == 0.88
        assert metrics.hit_rate == 0.95
        assert metrics.num_queries == 100
        assert metrics.k == 10

    def test_to_dict_default_k(self) -> None:
        """Test converting metrics to dict with default k."""
        metrics = RetrievalMetrics(
            recall_at_k=0.85,
            precision_at_k=0.90,
            mrr=0.75,
            ndcg_at_k=0.88,
            hit_rate=0.95,
            num_queries=100,
        )
        result = metrics.to_dict()
        assert result["recall@5"] == 0.85
        assert result["precision@5"] == 0.90
        assert result["mrr"] == 0.75
        assert result["ndcg@5"] == 0.88
        assert result["hit_rate"] == 0.95
        assert result["num_queries"] == 100

    def test_to_dict_custom_k(self) -> None:
        """Test converting metrics to dict with custom k."""
        metrics = RetrievalMetrics(
            recall_at_k=0.80,
            precision_at_k=0.88,
            k=10,
        )
        result = metrics.to_dict()
        assert "recall@10" in result
        assert "precision@10" in result
        assert result["recall@10"] == 0.80
        assert result["precision@10"] == 0.88

    def test_metric_bounds(self) -> None:
        """Test that metrics can have valid values."""
        metrics = RetrievalMetrics(
            recall_at_k=0.0,
            precision_at_k=1.0,
            mrr=0.5,
            ndcg_at_k=0.75,
            hit_rate=0.25,
        )
        assert metrics.recall_at_k >= 0.0
        assert metrics.precision_at_k <= 1.0


class TestQueryResult:
    """Test suite for QueryResult dataclass.

    Tests cover per-query result representation with retrieved document IDs,
    relevance judgments, and optional scores.
    """

    def test_initialization_minimal(self) -> None:
        """Test QueryResult with minimal fields."""
        result = QueryResult(query="test query")
        assert result.query == "test query"
        assert result.retrieved_ids == []
        assert result.retrieved_contents == []
        assert result.relevant_ids == set()
        assert result.scores == []

    def test_initialization_full(self) -> None:
        """Test QueryResult with all fields."""
        result = QueryResult(
            query="test query",
            retrieved_ids=["doc1", "doc2", "doc3"],
            retrieved_contents=["Content 1", "Content 2", "Content 3"],
            relevant_ids={"doc1", "doc3"},
            scores=[0.95, 0.85, 0.75],
        )
        assert result.query == "test query"
        assert result.retrieved_ids == ["doc1", "doc2", "doc3"]
        assert result.retrieved_contents == ["Content 1", "Content 2", "Content 3"]
        assert result.relevant_ids == {"doc1", "doc3"}
        assert result.scores == [0.95, 0.85, 0.75]

    def test_query_result_structure(self) -> None:
        """Test QueryResult maintains correct structure."""
        retrieved = ["doc1", "doc2"]
        contents = ["Content 1", "Content 2"]
        relevant = {"doc1"}
        scores = [0.95, 0.70]

        result = QueryResult(
            query="test",
            retrieved_ids=retrieved,
            retrieved_contents=contents,
            relevant_ids=relevant,
            scores=scores,
        )

        # Verify lists and sets are preserved
        assert isinstance(result.retrieved_ids, list)
        assert isinstance(result.relevant_ids, set)
        assert isinstance(result.scores, list)
        assert len(result.retrieved_ids) == 2
        assert len(result.relevant_ids) == 1

    def test_query_result_empty(self) -> None:
        """Test QueryResult with empty retrieval."""
        result = QueryResult(
            query="no results query",
            retrieved_ids=[],
            relevant_ids={"expected1", "expected2"},
        )
        assert result.query == "no results query"
        assert result.retrieved_ids == []
        assert len(result.relevant_ids) == 2


class TestEvaluationResult:
    """Test suite for EvaluationResult dataclass.

    Tests cover complete evaluation result serialization including pipeline
    name, dataset name, and aggregated metrics.
    """

    def test_to_dict_basic(self) -> None:
        """Test EvaluationResult.to_dict with basic values."""
        metrics = RetrievalMetrics(
            recall_at_k=0.8,
            precision_at_k=0.9,
            mrr=0.75,
            ndcg_at_k=0.85,
            hit_rate=0.95,
            num_queries=10,
            k=5,
        )
        result = EvaluationResult(
            metrics=metrics,
            pipeline_name="test_pipeline",
            dataset_name="test_dataset",
        )
        result_dict = result.to_dict()

        assert result_dict["pipeline"] == "test_pipeline"
        assert result_dict["dataset"] == "test_dataset"
        assert result_dict["num_queries"] == 0  # Based on query_results length
        assert "metrics" in result_dict
        assert result_dict["metrics"]["recall@5"] == 0.8

    def test_to_dict_with_query_results(self) -> None:
        """Test EvaluationResult.to_dict with query results."""
        metrics = RetrievalMetrics(k=10)
        query_results = [
            QueryResult(query="q1", retrieved_ids=["doc1"]),
            QueryResult(query="q2", retrieved_ids=["doc2"]),
            QueryResult(query="q3", retrieved_ids=["doc3"]),
        ]
        result = EvaluationResult(
            metrics=metrics,
            query_results=query_results,
            pipeline_name="pipeline",
            dataset_name="dataset",
        )
        result_dict = result.to_dict()

        assert result_dict["num_queries"] == 3
        assert "recall@10" in result_dict["metrics"]

    def test_to_dict_empty(self) -> None:
        """Test EvaluationResult.to_dict with defaults."""
        result = EvaluationResult(metrics=RetrievalMetrics())
        result_dict = result.to_dict()

        assert result_dict["pipeline"] == ""
        assert result_dict["dataset"] == ""
        assert result_dict["num_queries"] == 0


class TestComputeRecallAtK:
    """Test suite for compute_recall_at_k function.

    Recall@k measures what fraction of relevant documents appear in the
    top-k retrieved results. Formula: |relevant ∩ top-k| / |relevant|
    """

    def test_recall_perfect(self) -> None:
        """Test recall when all relevant docs are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_recall_partial(self) -> None:
        """Test recall when some relevant docs are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3", "doc5"}
        # 2 out of 3 relevant docs in top-3
        assert compute_recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_recall_none(self) -> None:
        """Test recall when no relevant docs are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_empty_relevant(self) -> None:
        """Test recall when there are no relevant docs (edge case)."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: set[str] = set()
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_empty_retrieved(self) -> None:
        """Test recall when no docs are retrieved."""
        retrieved: list[str] = []
        relevant = {"doc1", "doc2"}
        assert compute_recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_k_larger_than_retrieved(self) -> None:
        """Test recall when k is larger than retrieved list."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2", "doc3"}
        # 2 out of 3 relevant docs
        assert compute_recall_at_k(retrieved, relevant, k=10) == pytest.approx(2 / 3)

    def test_recall_k_smaller_than_retrieved(self) -> None:
        """Test recall with smaller k value."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc4"}
        # Only doc1 is in top-2
        assert compute_recall_at_k(retrieved, relevant, k=2) == 0.5

    def test_recall_various_k_values(self) -> None:
        """Test recall with different k values."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc5"}

        assert compute_recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)
        assert compute_recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
        assert compute_recall_at_k(retrieved, relevant, k=5) == 1.0


class TestComputePrecisionAtK:
    """Tests for compute_precision_at_k function."""

    def test_precision_perfect(self) -> None:
        """Test precision when all top-k docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        assert compute_precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_partial(self) -> None:
        """Test precision when some top-k docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3"}
        # 2 out of 3
        assert compute_precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_precision_none(self) -> None:
        """Test precision when no top-k docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert compute_precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_precision_k_zero(self) -> None:
        """Test precision when k=0 (edge case)."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        assert compute_precision_at_k(retrieved, relevant, k=0) == 0.0

    def test_precision_empty_retrieved(self) -> None:
        """Test precision when no docs are retrieved."""
        retrieved: list[str] = []
        relevant = {"doc1", "doc2"}
        assert compute_precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_precision_k_larger_than_retrieved(self) -> None:
        """Test precision when k > retrieved list size."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # 2 relevant out of k=5 (but only 2 retrieved)
        assert compute_precision_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 5)

    def test_precision_boundary_k_equals_one(self) -> None:
        """Test precision with k=1."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert compute_precision_at_k(retrieved, relevant, k=1) == 1.0

        relevant_miss = {"doc2"}
        assert compute_precision_at_k(retrieved, relevant_miss, k=1) == 0.0


class TestComputeMRR:
    """Tests for compute_mrr function."""

    def test_mrr_first_position(self) -> None:
        """Test MRR when relevant doc is at position 1."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert compute_mrr(retrieved, relevant) == 1.0

    def test_mrr_second_position(self) -> None:
        """Test MRR when relevant doc is at position 2."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}
        assert compute_mrr(retrieved, relevant) == 0.5

    def test_mrr_third_position(self) -> None:
        """Test MRR when relevant doc is at position 3."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc3"}
        assert compute_mrr(retrieved, relevant) == pytest.approx(1 / 3)

    def test_mrr_no_relevant_docs(self) -> None:
        """Test MRR when no relevant docs in retrieved list."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert compute_mrr(retrieved, relevant) == 0.0

    def test_mrr_empty_retrieved(self) -> None:
        """Test MRR when retrieved list is empty."""
        retrieved: list[str] = []
        relevant = {"doc1"}
        assert compute_mrr(retrieved, relevant) == 0.0

    def test_mrr_empty_relevant(self) -> None:
        """Test MRR when relevant set is empty."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: set[str] = set()
        assert compute_mrr(retrieved, relevant) == 0.0

    def test_mrr_multiple_relevant_returns_first(self) -> None:
        """Test MRR returns reciprocal of FIRST relevant doc position."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc2", "doc4"}  # Multiple relevant
        # First relevant is at position 2
        assert compute_mrr(retrieved, relevant) == 0.5

    def test_mrr_later_positions(self) -> None:
        """Test MRR at various later positions."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        assert compute_mrr(retrieved, {"doc4"}) == pytest.approx(1 / 4)
        assert compute_mrr(retrieved, {"doc5"}) == pytest.approx(1 / 5)


class TestComputeDCGAtK:
    """Tests for compute_dcg_at_k function."""

    def test_dcg_all_relevant(self) -> None:
        """Test DCG when all docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        expected = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        assert compute_dcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)

    def test_dcg_none_relevant(self) -> None:
        """Test DCG when no docs are relevant."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert compute_dcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_dcg_partial_relevant(self) -> None:
        """Test DCG with some relevant docs."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc3"}
        # DCG = 1/log2(2) + 0/log2(3) + 1/log2(4)
        expected = 1 / math.log2(2) + 1 / math.log2(4)
        assert compute_dcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)

    def test_dcg_empty_retrieved(self) -> None:
        """Test DCG when retrieved list is empty."""
        retrieved: list[str] = []
        relevant = {"doc1", "doc2"}
        assert compute_dcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_dcg_k_larger_than_retrieved(self) -> None:
        """Test DCG when k > retrieved list size."""
        retrieved = ["doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # Only 2 docs to consider
        expected = 1 / math.log2(2) + 1 / math.log2(3)
        assert compute_dcg_at_k(retrieved, relevant, k=10) == pytest.approx(expected)

    def test_dcg_single_doc(self) -> None:
        """Test DCG with single document."""
        retrieved = ["doc1"]
        relevant = {"doc1"}
        expected = 1 / math.log2(2)
        assert compute_dcg_at_k(retrieved, relevant, k=1) == pytest.approx(expected)


class TestComputeNDCGAtK:
    """Test suite for compute_ndcg_at_k function.

    NDCG normalizes DCG by the ideal DCG (all relevant docs at top).
    Returns a value in [0, 1] where 1.0 indicates perfect ranking.
    """

    def test_ndcg_perfect_ranking(self) -> None:
        """Test NDCG when ranking is perfect (all relevant at top)."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        # All relevant docs are at top - perfect NDCG
        assert compute_ndcg_at_k(retrieved, relevant, k=4) == 1.0

    def test_ndcg_worst_ranking(self) -> None:
        """Test NDCG when relevant docs are at bottom."""
        retrieved = ["doc3", "doc4", "doc1", "doc2"]
        relevant = {"doc1", "doc2"}
        # Relevant docs at positions 3 and 4
        # DCG = 1/log2(4) + 1/log2(5)
        # IDCG = 1/log2(2) + 1/log2(3)
        dcg = 1 / math.log2(4) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert compute_ndcg_at_k(retrieved, relevant, k=4) == pytest.approx(expected)

    def test_ndcg_no_relevant_docs(self) -> None:
        """Test NDCG when no relevant docs exist (idcg=0 case)."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: set[str] = set()
        # When there are no relevant docs, idcg=0, so ndcg=0
        assert compute_ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_ndcg_no_relevant_in_results(self) -> None:
        """Test NDCG when relevant docs not in retrieved list."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        # DCG = 0, IDCG > 0, so NDCG = 0
        assert compute_ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_ndcg_empty_retrieved(self) -> None:
        """Test NDCG when retrieved list is empty."""
        retrieved: list[str] = []
        relevant = {"doc1", "doc2"}
        # DCG = 0, IDCG > 0, but since no docs, result is 0
        assert compute_ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_ndcg_single_relevant(self) -> None:
        """Test NDCG with single relevant doc."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        # Perfect ranking for single doc
        assert compute_ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_ndcg_k_smaller_than_relevant(self) -> None:
        """Test NDCG when k < number of relevant docs."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3", "doc4", "doc5"}
        # k=2, only consider first 2 positions
        assert compute_ndcg_at_k(retrieved, relevant, k=2) == 1.0


class TestComputeHitRate:
    """Test suite for compute_hit_rate function.

    Hit rate is a binary metric returning 1.0 if any relevant document
    appears in top-k, 0.0 otherwise.
    """

    def test_hit_rate_hit(self) -> None:
        """Test hit rate when there's a hit."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}
        assert compute_hit_rate(retrieved, relevant, k=3) == 1.0

    def test_hit_rate_miss(self) -> None:
        """Test hit rate when there's no hit."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4", "doc5"}
        assert compute_hit_rate(retrieved, relevant, k=3) == 0.0

    def test_hit_rate_first_position(self) -> None:
        """Test hit rate with relevant doc at first position."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert compute_hit_rate(retrieved, relevant, k=1) == 1.0

    def test_hit_rate_outside_k(self) -> None:
        """Test hit rate when relevant doc is outside top-k."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc5"}
        assert compute_hit_rate(retrieved, relevant, k=3) == 0.0
        assert compute_hit_rate(retrieved, relevant, k=5) == 1.0

    def test_hit_rate_empty_retrieved(self) -> None:
        """Test hit rate with empty retrieved list."""
        retrieved: list[str] = []
        relevant = {"doc1"}
        assert compute_hit_rate(retrieved, relevant, k=3) == 0.0

    def test_hit_rate_empty_relevant(self) -> None:
        """Test hit rate with empty relevant set."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant: set[str] = set()
        assert compute_hit_rate(retrieved, relevant, k=3) == 0.0

    def test_hit_rate_multiple_relevant(self) -> None:
        """Test hit rate with multiple relevant docs (still binary)."""
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2", "doc3"}
        # Hit rate is binary - 1 if ANY relevant doc in top-k
        assert compute_hit_rate(retrieved, relevant, k=3) == 1.0


class TestEvaluateRetrieval:
    """Test suite for evaluate_retrieval function.

    Tests the main evaluation entry point that computes all metrics
    from a list of QueryResult objects and returns aggregated RetrievalMetrics.
    """

    def test_evaluate_empty_results(self) -> None:
        """Test evaluate_retrieval with empty query results."""
        metrics = evaluate_retrieval([], k=5)

        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg_at_k == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.num_queries == 0
        assert metrics.k == 5

    def test_evaluate_single_query_perfect(self) -> None:
        """Test evaluate_retrieval with single perfect query."""
        query_results = [
            QueryResult(
                query="test query",
                retrieved_ids=["doc1", "doc2", "doc3"],
                relevant_ids={"doc1", "doc2", "doc3"},
            )
        ]
        metrics = evaluate_retrieval(query_results, k=3)

        assert metrics.recall_at_k == 1.0
        assert metrics.precision_at_k == 1.0
        assert metrics.mrr == 1.0
        assert metrics.ndcg_at_k == 1.0
        assert metrics.hit_rate == 1.0
        assert metrics.num_queries == 1
        assert metrics.k == 3

    def test_evaluate_single_query_no_hits(self) -> None:
        """Test evaluate_retrieval with single query with no hits."""
        query_results = [
            QueryResult(
                query="test query",
                retrieved_ids=["doc1", "doc2", "doc3"],
                relevant_ids={"doc4", "doc5"},
            )
        ]
        metrics = evaluate_retrieval(query_results, k=3)

        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.mrr == 0.0
        assert metrics.ndcg_at_k == 0.0
        assert metrics.hit_rate == 0.0
        assert metrics.num_queries == 1

    def test_evaluate_multiple_queries_aggregation(self) -> None:
        """Test evaluate_retrieval aggregates multiple queries correctly."""
        query_results = [
            # Perfect result
            QueryResult(
                query="q1",
                retrieved_ids=["doc1", "doc2"],
                relevant_ids={"doc1", "doc2"},
            ),
            # No hits
            QueryResult(
                query="q2",
                retrieved_ids=["doc3", "doc4"],
                relevant_ids={"doc5", "doc6"},
            ),
        ]
        metrics = evaluate_retrieval(query_results, k=2)

        # Average of 1.0 and 0.0 = 0.5 for most metrics
        assert metrics.recall_at_k == pytest.approx(0.5)
        assert metrics.precision_at_k == pytest.approx(0.5)
        assert metrics.mrr == pytest.approx(0.5)
        assert metrics.hit_rate == pytest.approx(0.5)
        assert metrics.num_queries == 2

    def test_evaluate_different_k_values(self) -> None:
        """Test evaluate_retrieval with different k values."""
        query_results = [
            QueryResult(
                query="test",
                retrieved_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
                relevant_ids={"doc1", "doc5"},
            )
        ]

        metrics_k1 = evaluate_retrieval(query_results, k=1)
        metrics_k5 = evaluate_retrieval(query_results, k=5)

        # With k=1, only doc1 is considered
        assert metrics_k1.recall_at_k == 0.5  # 1/2 relevant found
        assert metrics_k1.precision_at_k == 1.0  # 1/1 in top-k are relevant

        # With k=5, both doc1 and doc5 are considered
        assert metrics_k5.recall_at_k == 1.0  # 2/2 relevant found
        assert metrics_k5.precision_at_k == 0.4  # 2/5 in top-k are relevant

    def test_evaluate_partial_relevance(self) -> None:
        """Test evaluate_retrieval with partial relevance."""
        query_results = [
            QueryResult(
                query="test",
                retrieved_ids=["doc1", "doc2", "doc3"],
                relevant_ids={"doc2", "doc3", "doc4"},  # doc4 not retrieved
            )
        ]
        metrics = evaluate_retrieval(query_results, k=3)

        # 2 out of 3 relevant docs retrieved
        assert metrics.recall_at_k == pytest.approx(2 / 3)
        # 2 out of 3 top-k are relevant
        assert metrics.precision_at_k == pytest.approx(2 / 3)
        # First relevant at position 2
        assert metrics.mrr == pytest.approx(0.5)
        # Hit rate is 1.0 (at least one relevant in top-k)
        assert metrics.hit_rate == 1.0

    def test_evaluate_retrieval_default_k(self) -> None:
        """Test evaluate_retrieval uses default k=5."""
        query_results = [
            QueryResult(
                query="test",
                retrieved_ids=["doc1"],
                relevant_ids={"doc1"},
            )
        ]
        metrics = evaluate_retrieval(query_results)  # No k specified
        assert metrics.k == 5
