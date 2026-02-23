"""Tests for RetrievalMetrics and MetricsAggregator classes."""

import math

import pytest

from vectordb.haystack.cost_optimized_rag.base.metrics import (
    MetricsAggregator,
    RetrievalMetrics,
)


class TestRetrievalMetricsMRR:
    """Tests for RetrievalMetrics.mrr() method."""

    def test_mrr_found_at_rank_1(self) -> None:
        """Test MRR when relevant doc is at rank 1."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert RetrievalMetrics.mrr(retrieved, relevant) == 1.0

    def test_mrr_found_at_rank_2(self) -> None:
        """Test MRR when relevant doc is at rank 2."""
        retrieved = ["a", "b", "c"]
        relevant = {"b"}
        assert RetrievalMetrics.mrr(retrieved, relevant) == 0.5

    def test_mrr_found_at_rank_3(self) -> None:
        """Test MRR when relevant doc is at rank 3."""
        retrieved = ["a", "b", "c"]
        relevant = {"c"}
        assert RetrievalMetrics.mrr(retrieved, relevant) == pytest.approx(1.0 / 3.0)

    def test_mrr_found_at_rank_5(self) -> None:
        """Test MRR when relevant doc is at rank 5."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"e"}
        assert RetrievalMetrics.mrr(retrieved, relevant) == 0.2

    def test_mrr_not_found(self) -> None:
        """Test MRR returns 0 when no relevant doc found."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        assert RetrievalMetrics.mrr(retrieved, relevant) == 0.0

    def test_mrr_empty_retrieved(self) -> None:
        """Test MRR with empty retrieved list."""
        retrieved: list[str] = []
        relevant = {"a"}
        assert RetrievalMetrics.mrr(retrieved, relevant) == 0.0

    def test_mrr_empty_relevant(self) -> None:
        """Test MRR with empty relevant set."""
        retrieved = ["a", "b", "c"]
        relevant: set[str] = set()
        assert RetrievalMetrics.mrr(retrieved, relevant) == 0.0

    def test_mrr_multiple_relevant_first_match(self) -> None:
        """Test MRR returns 1/rank of FIRST relevant doc found."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"c", "d"}
        # First match is at rank 3
        assert RetrievalMetrics.mrr(retrieved, relevant) == pytest.approx(1.0 / 3.0)

    def test_mrr_both_empty(self) -> None:
        """Test MRR with both empty inputs."""
        assert RetrievalMetrics.mrr([], set()) == 0.0


class TestRetrievalMetricsRecallAtK:
    """Tests for RetrievalMetrics.recall_at_k() method."""

    def test_recall_at_k_all_found(self) -> None:
        """Test recall@K when all relevant docs are in top-K."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b"}
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_partial_found(self) -> None:
        """Test recall@K with partial matches."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "x", "y"}
        # 2 out of 4 found in top 5
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=5) == 0.5

    def test_recall_at_k_none_found(self) -> None:
        """Test recall@K when no relevant docs found."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y", "z"}
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_at_k_no_relevant_documents(self) -> None:
        """Test recall@K with empty relevant set returns 0."""
        retrieved = ["a", "b", "c"]
        relevant: set[str] = set()
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_recall_at_k_cutoff(self) -> None:
        """Test recall@K respects K cutoff."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "e"}
        # k=3: only "a" found, so recall = 1/2
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=3) == 0.5
        # k=5: both found, recall = 2/2
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_recall_at_k_various_k_values(self) -> None:
        """Test recall@K with various K values."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"b", "c", "d"}
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=1) == 0.0
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=2) == pytest.approx(
            1 / 3
        )
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=3) == pytest.approx(
            2 / 3
        )
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=4) == 1.0

    def test_recall_at_k_default_k(self) -> None:
        """Test recall@K uses default k=10."""
        retrieved = list("abcdefghijk")  # 11 items
        relevant = {"k"}
        # Default k=10, "k" is at position 11, not in top 10
        assert RetrievalMetrics.recall_at_k(retrieved, relevant) == 0.0

    def test_recall_at_k_k_larger_than_retrieved(self) -> None:
        """Test recall@K when K is larger than retrieved list."""
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        # k=10, but only 2 docs retrieved, 2 out of 3 found
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=10) == pytest.approx(
            2 / 3
        )


class TestRetrievalMetricsPrecisionAtK:
    """Tests for RetrievalMetrics.precision_at_k() method."""

    def test_precision_at_k_all_matches(self) -> None:
        """Test precision@K when all retrieved are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_at_k_no_matches(self) -> None:
        """Test precision@K when no retrieved are relevant."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y", "z"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_precision_at_k_partial_matches(self) -> None:
        """Test precision@K with partial matches."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        # 2 out of 4
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_precision_at_k_zero(self) -> None:
        """Test precision@K returns 0 when k=0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=0) == 0.0

    def test_precision_at_k_negative(self) -> None:
        """Test precision@K returns 0 when k<0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=-1) == 0.0

    def test_precision_at_k_various_k(self) -> None:
        """Test precision@K with various K values."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "d"}
        # k=1: 1 relevant in top 1 = 1/1 = 1.0
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=1) == 1.0
        # k=2: 1 relevant in top 2 = 1/2 = 0.5
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=2) == 0.5
        # k=3: 1 relevant in top 3 = 1/3
        assert RetrievalMetrics.precision_at_k(
            retrieved, relevant, k=3
        ) == pytest.approx(1 / 3)
        # k=4: 2 relevant in top 4 = 2/4 = 0.5
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_precision_at_k_k_larger_than_retrieved(self) -> None:
        """Test precision@K when K > len(retrieved)."""
        retrieved = ["a", "b"]
        relevant = {"a", "b"}
        # k=5, 2 relevant found, precision = 2/5 = 0.4
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=5) == 0.4


class TestRetrievalMetricsNDCGAtK:
    """Tests for RetrievalMetrics.ndcg_at_k() method."""

    def test_ndcg_at_k_perfect_ranking(self) -> None:
        """Test NDCG@K with perfect ranking (all relevant at top)."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        # Perfect: both relevant docs at positions 1 and 2
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=4) == 1.0

    def test_ndcg_at_k_no_relevant(self) -> None:
        """Test NDCG@K with no relevant documents."""
        retrieved = ["a", "b", "c"]
        relevant: set[str] = set()
        # IDCG = 0, so should return 0
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_ndcg_at_k_none_found(self) -> None:
        """Test NDCG@K when relevant docs not in retrieved."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        # DCG = 0, IDCG > 0, so NDCG = 0
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3) == 0.0

    def test_ndcg_at_k_partial_matches(self) -> None:
        """Test NDCG@K with partial matches in suboptimal order."""
        retrieved = ["x", "a", "y", "b"]
        relevant = {"a", "b"}
        k = 4
        # DCG: 0/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5)
        dcg = 0 + 1 / math.log2(3) + 0 + 1 / math.log2(5)
        # IDCG: 1/log2(2) + 1/log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=k) == pytest.approx(
            expected
        )

    def test_ndcg_at_k_various_k_values(self) -> None:
        """Test NDCG@K with various K values."""
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}

        # k=1: DCG = 1/log2(2) = 1, IDCG = 1/log2(2) = 1
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=1) == 1.0

        # k=2: DCG = 1/log2(2) + 0 = 1, IDCG = 1/log2(2) + 1/log2(3)
        dcg_2 = 1 / math.log2(2)
        idcg_2 = 1 / math.log2(2) + 1 / math.log2(3)
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(
            dcg_2 / idcg_2
        )

    def test_ndcg_at_k_single_relevant_at_end(self) -> None:
        """Test NDCG@K with single relevant doc at end."""
        retrieved = ["x", "y", "z", "a"]
        relevant = {"a"}
        k = 4
        # DCG: 0 + 0 + 0 + 1/log2(5)
        dcg = 1 / math.log2(5)
        # IDCG: 1/log2(2)
        idcg = 1 / math.log2(2)
        expected = dcg / idcg
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=k) == pytest.approx(
            expected
        )


class TestRetrievalMetricsMAPAtK:
    """Tests for RetrievalMetrics.map_at_k() method."""

    def test_map_at_k_multiple_relevant(self) -> None:
        """Test MAP@K with multiple relevant docs."""
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        k = 4
        # At pos 1 ("a" is relevant): precision@1 = 1/1 = 1.0
        # At pos 3 ("b" is relevant): precision@3 = 2/3
        # MAP = (1.0 + 2/3) / min(4, 2) = (1.0 + 0.666) / 2
        expected = (1.0 + 2 / 3) / 2
        assert RetrievalMetrics.map_at_k(retrieved, relevant, k=k) == pytest.approx(
            expected
        )

    def test_map_at_k_no_relevant(self) -> None:
        """Test MAP@K with empty relevant set."""
        retrieved = ["a", "b", "c"]
        relevant: set[str] = set()
        assert RetrievalMetrics.map_at_k(retrieved, relevant, k=3) == 0.0

    def test_map_at_k_none_found(self) -> None:
        """Test MAP@K when relevant docs not in retrieved."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y"}
        # precision_sum = 0, result = 0 / min(3, 2) = 0
        assert RetrievalMetrics.map_at_k(retrieved, relevant, k=3) == 0.0

    def test_map_at_k_perfect_ranking(self) -> None:
        """Test MAP@K with perfect ranking."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        k = 4
        # At pos 1: precision@1 = 1/1 = 1.0
        # At pos 2: precision@2 = 2/2 = 1.0
        # MAP = (1.0 + 1.0) / min(4, 2) = 2.0 / 2 = 1.0
        assert RetrievalMetrics.map_at_k(retrieved, relevant, k=k) == 1.0

    def test_map_at_k_single_relevant_at_start(self) -> None:
        """Test MAP@K with single relevant at start."""
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        k = 3
        # At pos 1: precision@1 = 1.0
        # MAP = 1.0 / min(3, 1) = 1.0
        assert RetrievalMetrics.map_at_k(retrieved, relevant, k=k) == 1.0

    def test_map_at_k_single_relevant_at_end(self) -> None:
        """Test MAP@K with single relevant at end of k window."""
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        k = 3
        # At pos 3: precision@3 = 1/3
        # MAP = (1/3) / min(3, 1) = 1/3
        assert RetrievalMetrics.map_at_k(retrieved, relevant, k=k) == pytest.approx(
            1 / 3
        )


class TestRetrievalMetricsComputeAll:
    """Tests for RetrievalMetrics.compute_all() method."""

    def test_compute_all_returns_all_metrics(self) -> None:
        """Test compute_all returns dict with all metrics."""
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        result = RetrievalMetrics.compute_all(retrieved, relevant, k=3)

        assert isinstance(result, dict)
        assert "mrr" in result
        assert "recall_at_k" in result
        assert "precision_at_k" in result
        assert "ndcg_at_k" in result
        assert "map_at_k" in result

    def test_compute_all_values_match_individual(self) -> None:
        """Test compute_all values match individual method calls."""
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        k = 5

        result = RetrievalMetrics.compute_all(retrieved, relevant, k)

        assert result["mrr"] == RetrievalMetrics.mrr(retrieved, relevant)
        assert result["recall_at_k"] == RetrievalMetrics.recall_at_k(
            retrieved, relevant, k
        )
        assert result["precision_at_k"] == RetrievalMetrics.precision_at_k(
            retrieved, relevant, k
        )
        assert result["ndcg_at_k"] == RetrievalMetrics.ndcg_at_k(retrieved, relevant, k)
        assert result["map_at_k"] == RetrievalMetrics.map_at_k(retrieved, relevant, k)

    def test_compute_all_empty_inputs(self) -> None:
        """Test compute_all with empty inputs."""
        result = RetrievalMetrics.compute_all([], set(), k=10)

        assert result["mrr"] == 0.0
        assert result["recall_at_k"] == 0.0
        assert result["precision_at_k"] == 0.0
        assert result["ndcg_at_k"] == 0.0
        assert result["map_at_k"] == 0.0

    def test_compute_all_perfect_retrieval(self) -> None:
        """Test compute_all with perfect retrieval."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        result = RetrievalMetrics.compute_all(retrieved, relevant, k=3)

        assert result["mrr"] == 1.0
        assert result["recall_at_k"] == 1.0
        assert result["precision_at_k"] == 1.0
        assert result["ndcg_at_k"] == 1.0
        assert result["map_at_k"] == 1.0


class TestMetricsAggregatorInit:
    """Tests for MetricsAggregator.__init__() method."""

    def test_init_creates_empty_results(self) -> None:
        """Test initialization creates empty results list."""
        aggregator = MetricsAggregator()
        assert aggregator.results == []

    def test_init_results_is_list(self) -> None:
        """Test results attribute is a list."""
        aggregator = MetricsAggregator()
        assert isinstance(aggregator.results, list)


class TestMetricsAggregatorAddResult:
    """Tests for MetricsAggregator.add_result() method."""

    def test_add_result_single(self) -> None:
        """Test adding a single result."""
        aggregator = MetricsAggregator()
        aggregator.add_result(
            query_id="q1",
            retrieved_ids=["a", "b", "c"],
            relevant_ids={"a"},
            k=3,
        )

        assert len(aggregator.results) == 1
        assert aggregator.results[0]["query_id"] == "q1"
        assert aggregator.results[0]["retrieved_count"] == 3
        assert aggregator.results[0]["relevant_count"] == 1

    def test_add_result_multiple(self) -> None:
        """Test adding multiple results."""
        aggregator = MetricsAggregator()
        aggregator.add_result("q1", ["a", "b"], {"a"}, k=2)
        aggregator.add_result("q2", ["x", "y", "z"], {"x", "y"}, k=3)
        aggregator.add_result("q3", ["p"], {"p"}, k=1)

        assert len(aggregator.results) == 3
        assert aggregator.results[0]["query_id"] == "q1"
        assert aggregator.results[1]["query_id"] == "q2"
        assert aggregator.results[2]["query_id"] == "q3"

    def test_add_result_computes_metrics(self) -> None:
        """Test add_result computes all metrics."""
        aggregator = MetricsAggregator()
        aggregator.add_result(
            query_id="q1",
            retrieved_ids=["a", "b"],
            relevant_ids={"a"},
            k=2,
        )

        result = aggregator.results[0]
        assert "mrr" in result
        assert "recall_at_k" in result
        assert "precision_at_k" in result
        assert "ndcg_at_k" in result
        assert "map_at_k" in result
        assert result["mrr"] == 1.0  # "a" at rank 1

    def test_add_result_default_k(self) -> None:
        """Test add_result uses default k=10."""
        aggregator = MetricsAggregator()
        aggregator.add_result(
            query_id="q1",
            retrieved_ids=list("abcdefghijk"),  # 11 items
            relevant_ids={"k"},  # at position 11
        )

        result = aggregator.results[0]
        # "k" is at position 11, not in top 10, so recall = 0
        assert result["recall_at_k"] == 0.0


class TestMetricsAggregatorAggregate:
    """Tests for MetricsAggregator.aggregate() method."""

    def test_aggregate_single_result(self) -> None:
        """Test aggregate with single result."""
        aggregator = MetricsAggregator()
        aggregator.add_result("q1", ["a", "b", "c"], {"a"}, k=3)

        agg = aggregator.aggregate()

        assert "avg_mrr" in agg
        assert "avg_recall_at_k" in agg
        assert "avg_precision_at_k" in agg
        assert "avg_ndcg_at_k" in agg
        assert "avg_map_at_k" in agg
        # Single result, average equals the value
        assert agg["avg_mrr"] == 1.0

    def test_aggregate_multiple_results(self) -> None:
        """Test aggregate averages across multiple results."""
        aggregator = MetricsAggregator()
        # q1: mrr=1.0 (a at rank 1)
        aggregator.add_result("q1", ["a", "b", "c"], {"a"}, k=3)
        # q2: mrr=0.5 (b at rank 2)
        aggregator.add_result("q2", ["x", "b", "c"], {"b"}, k=3)

        agg = aggregator.aggregate()

        # avg_mrr = (1.0 + 0.5) / 2 = 0.75
        assert agg["avg_mrr"] == pytest.approx(0.75)

    def test_aggregate_no_results_raises_error(self) -> None:
        """Test aggregate raises ValueError when no results."""
        aggregator = MetricsAggregator()

        with pytest.raises(ValueError, match="No results to aggregate"):
            aggregator.aggregate()

    def test_aggregate_returns_dict(self) -> None:
        """Test aggregate returns a dictionary."""
        aggregator = MetricsAggregator()
        aggregator.add_result("q1", ["a"], {"a"}, k=1)

        agg = aggregator.aggregate()

        assert isinstance(agg, dict)
        assert len(agg) == 5  # 5 metrics

    def test_aggregate_all_zero_metrics(self) -> None:
        """Test aggregate with all zero metrics."""
        aggregator = MetricsAggregator()
        aggregator.add_result("q1", ["a", "b"], {"x", "y"}, k=2)
        aggregator.add_result("q2", ["c", "d"], {"z"}, k=2)

        agg = aggregator.aggregate()

        assert agg["avg_mrr"] == 0.0
        assert agg["avg_recall_at_k"] == 0.0
        assert agg["avg_precision_at_k"] == 0.0


class TestMetricsAggregatorSummary:
    """Tests for MetricsAggregator.summary() method."""

    def test_summary_includes_total_queries(self) -> None:
        """Test summary includes total_queries count."""
        aggregator = MetricsAggregator()
        aggregator.add_result("q1", ["a"], {"a"}, k=1)
        aggregator.add_result("q2", ["b"], {"b"}, k=1)
        aggregator.add_result("q3", ["c"], {"c"}, k=1)

        summary = aggregator.summary()

        assert summary["total_queries"] == 3

    def test_summary_includes_all_avg_metrics(self) -> None:
        """Test summary includes all average metrics."""
        aggregator = MetricsAggregator()
        aggregator.add_result("q1", ["a", "b"], {"a"}, k=2)

        summary = aggregator.summary()

        assert "total_queries" in summary
        assert "avg_mrr" in summary
        assert "avg_recall_at_k" in summary
        assert "avg_precision_at_k" in summary
        assert "avg_ndcg_at_k" in summary
        assert "avg_map_at_k" in summary

    def test_summary_empty_aggregator(self) -> None:
        """Test summary with no results returns minimal dict."""
        aggregator = MetricsAggregator()

        summary = aggregator.summary()

        assert summary == {"total_queries": 0}

    def test_summary_correct_values(self) -> None:
        """Test summary returns correct aggregated values."""
        aggregator = MetricsAggregator()
        # Perfect retrieval
        aggregator.add_result("q1", ["a", "b"], {"a", "b"}, k=2)
        # No relevant found
        aggregator.add_result("q2", ["c", "d"], {"x", "y"}, k=2)

        summary = aggregator.summary()

        assert summary["total_queries"] == 2
        # avg_mrr = (1.0 + 0.0) / 2 = 0.5
        assert summary["avg_mrr"] == pytest.approx(0.5)
        # avg_recall_at_k = (1.0 + 0.0) / 2 = 0.5
        assert summary["avg_recall_at_k"] == pytest.approx(0.5)

    def test_summary_is_complete(self) -> None:
        """Test summary contains exactly 6 keys when results exist."""
        aggregator = MetricsAggregator()
        aggregator.add_result("q1", ["a"], {"a"}, k=1)

        summary = aggregator.summary()

        expected_keys = {
            "total_queries",
            "avg_mrr",
            "avg_recall_at_k",
            "avg_precision_at_k",
            "avg_ndcg_at_k",
            "avg_map_at_k",
        }
        assert set(summary.keys()) == expected_keys
