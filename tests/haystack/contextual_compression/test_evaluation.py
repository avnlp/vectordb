"""Comprehensive tests for compression evaluation module.

Tests cover:
- CompressionEvaluationMetrics dataclass
- CompressionEvaluator with NDCG, MRR, Recall@K calculations
- print_detailed_report function
"""

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.compression_utils import RankerResult
from vectordb.haystack.contextual_compression.evaluation import (
    CompressionEvaluationMetrics,
    CompressionEvaluator,
    print_detailed_report,
)


class TestCompressionEvaluationMetrics:
    """Tests for CompressionEvaluationMetrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test creating metrics with all fields."""
        metrics = CompressionEvaluationMetrics(
            ndcg=0.85,
            mrr=0.90,
            recall_at_k={1: 0.7, 5: 0.85, 10: 0.90},
            tokens_saved=1000,
            compression_ratio=0.5,
            mean_score=0.88,
        )

        assert metrics.ndcg == 0.85
        assert metrics.mrr == 0.90
        assert metrics.recall_at_k == {1: 0.7, 5: 0.85, 10: 0.90}
        assert metrics.tokens_saved == 1000
        assert metrics.compression_ratio == 0.5
        assert metrics.mean_score == 0.88

    def test_metrics_default_values(self) -> None:
        """Test metrics with zero/empty values."""
        metrics = CompressionEvaluationMetrics(
            ndcg=0.0,
            mrr=0.0,
            recall_at_k={},
            tokens_saved=0,
            compression_ratio=1.0,
            mean_score=0.0,
        )

        assert metrics.ndcg == 0.0
        assert metrics.compression_ratio == 1.0


class TestCompressionEvaluatorCalculateNDCG:
    """Tests for CompressionEvaluator.calculate_ndcg method."""

    def test_ndcg_perfect_ranking(self) -> None:
        """Test NDCG with perfect ranking (all relevant docs at top)."""
        # Create ideal results
        ideal_docs = [
            Document(content="doc1", id="1"),
            Document(content="doc2", id="2"),
            Document(content="doc3", id="3"),
        ]

        # Ranked results match ideal exactly
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
            RankerResult(document=Document(content="doc3", id="3"), score=0.7),
        ]

        ndcg = CompressionEvaluator.calculate_ndcg(ranked_results, ideal_docs, k=3)
        assert ndcg == 1.0

    def test_ndcg_partial_match(self) -> None:
        """Test NDCG with partial relevance match."""
        ideal_docs = [
            Document(content="doc1", id="1"),
            Document(content="doc2", id="2"),
        ]

        # Only first result is relevant
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="docX", id="X"), score=0.8),
            RankerResult(document=Document(content="docY", id="Y"), score=0.7),
        ]

        ndcg = CompressionEvaluator.calculate_ndcg(ranked_results, ideal_docs, k=3)
        # Using standard NDCG formula with log2 discount:
        # actual_relevances = [1.0, 0.0, 0.0]
        # ideal_relevances = [1.0, 1.0, 0.0]
        # actual_dcg = 1/log2(2) + 0/log2(3) + 0/log2(4) = 1.0
        # ideal_dcg = 1/log2(2) + 1/log2(3) + 0/log2(4) = 1.0 + 0.631 = 1.631
        # NDCG = 1.0 / 1.631 = 0.613
        assert ndcg == pytest.approx(0.613, abs=0.01)

    def test_ndcg_no_match(self) -> None:
        """Test NDCG when no documents match."""
        ideal_docs = [Document(content="doc1", id="1")]

        ranked_results = [
            RankerResult(document=Document(content="docX", id="X"), score=0.9),
            RankerResult(document=Document(content="docY", id="Y"), score=0.8),
        ]

        ndcg = CompressionEvaluator.calculate_ndcg(ranked_results, ideal_docs, k=2)
        assert ndcg == 0.0

    def test_ndcg_empty_results(self) -> None:
        """Test NDCG with empty results."""
        ideal_docs = [Document(content="doc1", id="1")]

        ndcg = CompressionEvaluator.calculate_ndcg([], ideal_docs, k=5)
        assert ndcg == 0.0

    def test_ndcg_empty_ideal(self) -> None:
        """Test NDCG with empty ideal results."""
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
        ]

        ndcg = CompressionEvaluator.calculate_ndcg(ranked_results, [], k=5)
        assert ndcg == 0.0

    def test_ndcg_both_empty(self) -> None:
        """Test NDCG with both empty."""
        ndcg = CompressionEvaluator.calculate_ndcg([], [], k=5)
        assert ndcg == 0.0

    def test_ndcg_respects_k(self) -> None:
        """Test that NDCG respects the k parameter.

        Note: The implementation calculates ideal DCG based on min(len(ideal), k)
        and actual DCG based on ranked_results[:k]. If the relevant doc is not
        in the top-k of ranked_results, the actual DCG will be lower.
        """
        ideal_docs = [Document(content="doc1", id="1")]

        # Create ranked results with relevant doc at position 1 (index 0)
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
            RankerResult(document=Document(content="doc3", id="3"), score=0.7),
            RankerResult(document=Document(content="doc4", id="4"), score=0.6),
            RankerResult(document=Document(content="doc5", id="5"), score=0.5),
        ]

        # With k=1, only first doc is considered
        ndcg_k1 = CompressionEvaluator.calculate_ndcg(ranked_results, ideal_docs, k=1)
        # ideal_relevances = [1.0]
        # actual_relevances = [1.0] (doc1 is at position 1)
        # Ideal DCG = 1/2 = 0.5
        # Actual DCG = 1/2 = 0.5
        # NDCG = 0.5 / 0.5 = 1.0
        assert ndcg_k1 == 1.0

        # With k=3, top 3 docs are considered
        ndcg_k3 = CompressionEvaluator.calculate_ndcg(ranked_results, ideal_docs, k=3)
        # ideal_relevances = [1.0, 0.0, 0.0]
        # actual_relevances = [1.0, 0.0, 0.0]
        # ideal_dcg = 1/log2(2) = 1.0
        # actual_dcg = 1/log2(2) = 1.0
        # NDCG = 1.0 / 1.0 = 1.0
        assert ndcg_k3 == 1.0

        # Now test with relevant doc NOT in top-k
        ranked_results_reordered = [
            RankerResult(document=Document(content="doc2", id="2"), score=0.9),
            RankerResult(document=Document(content="doc3", id="3"), score=0.8),
            RankerResult(document=Document(content="doc4", id="4"), score=0.7),
            RankerResult(document=Document(content="doc5", id="5"), score=0.6),
            RankerResult(document=Document(content="doc1", id="1"), score=0.5),
        ]

        # With k=3, relevant doc (doc1) is NOT in top 3
        ndcg_k3_missing = CompressionEvaluator.calculate_ndcg(
            ranked_results_reordered, ideal_docs, k=3
        )
        # ideal_relevances = [1.0, 0.0, 0.0]
        # actual_relevances = [0.0, 0.0, 0.0] (doc1 is at position 5, not in top 3)
        # ideal_dcg = 1/log2(2) = 1.0
        # actual_dcg = 0
        # NDCG = 0 / 1.0 = 0
        assert ndcg_k3_missing == 0.0

        # With k=5, relevant doc is in top-k at position 5
        ndcg_k5 = CompressionEvaluator.calculate_ndcg(
            ranked_results_reordered, ideal_docs, k=5
        )
        # ideal_relevances = [1.0, 0.0, 0.0, 0.0, 0.0]
        # actual_relevances = [0.0, 0.0, 0.0, 0.0, 1.0]
        # ideal_dcg = 1/log2(2) = 1.0
        # actual_dcg = 1/log2(6) = 0.387
        # NDCG = 0.387 / 1.0 = 0.387
        assert ndcg_k5 == pytest.approx(0.387, abs=0.01)

    def test_ndcg_with_none_ids(self) -> None:
        """Test NDCG handling documents with None IDs."""
        ideal_docs = [Document(content="doc1", id="1")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id=None), score=0.8),
        ]

        ndcg = CompressionEvaluator.calculate_ndcg(ranked_results, ideal_docs, k=2)
        assert ndcg > 0.0


class TestCompressionEvaluatorCalculateMRR:
    """Tests for CompressionEvaluator.calculate_mrr method."""

    def test_mrr_first_position(self) -> None:
        """Test MRR when relevant doc is at position 1."""
        relevant_docs = [Document(content="doc1", id="1")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
        ]

        mrr = CompressionEvaluator.calculate_mrr(ranked_results, relevant_docs)
        assert mrr == 1.0

    def test_mrr_second_position(self) -> None:
        """Test MRR when relevant doc is at position 2."""
        relevant_docs = [Document(content="doc2", id="2")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
        ]

        mrr = CompressionEvaluator.calculate_mrr(ranked_results, relevant_docs)
        assert mrr == 0.5

    def test_mrr_third_position(self) -> None:
        """Test MRR when relevant doc is at position 3."""
        relevant_docs = [Document(content="doc3", id="3")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
            RankerResult(document=Document(content="doc3", id="3"), score=0.7),
        ]

        mrr = CompressionEvaluator.calculate_mrr(ranked_results, relevant_docs)
        assert mrr == pytest.approx(1.0 / 3.0)

    def test_mrr_not_found(self) -> None:
        """Test MRR when no relevant doc is found."""
        relevant_docs = [Document(content="docX", id="X")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
        ]

        mrr = CompressionEvaluator.calculate_mrr(ranked_results, relevant_docs)
        assert mrr == 0.0

    def test_mrr_empty_ranked(self) -> None:
        """Test MRR with empty ranked results."""
        relevant_docs = [Document(content="doc1", id="1")]

        mrr = CompressionEvaluator.calculate_mrr([], relevant_docs)
        assert mrr == 0.0

    def test_mrr_empty_relevant(self) -> None:
        """Test MRR with empty relevant docs."""
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
        ]

        mrr = CompressionEvaluator.calculate_mrr(ranked_results, [])
        assert mrr == 0.0

    def test_mrr_multiple_relevant_first_match(self) -> None:
        """Test MRR returns reciprocal of FIRST relevant doc found."""
        relevant_docs = [
            Document(content="doc2", id="2"),
            Document(content="doc3", id="3"),
        ]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
            RankerResult(document=Document(content="doc3", id="3"), score=0.7),
        ]

        mrr = CompressionEvaluator.calculate_mrr(ranked_results, relevant_docs)
        # First match is at position 2
        assert mrr == 0.5

    def test_mrr_with_none_ids(self) -> None:
        """Test MRR handling documents with None IDs."""
        relevant_docs = [Document(content="doc1", id="1")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id=None), score=0.8),
        ]

        mrr = CompressionEvaluator.calculate_mrr(ranked_results, relevant_docs)
        assert mrr == 1.0


class TestCompressionEvaluatorCalculateRecallAtK:
    """Tests for CompressionEvaluator.calculate_recall_at_k method."""

    def test_recall_at_k_all_found(self) -> None:
        """Test recall when all relevant docs are in top-k."""
        relevant_docs = [
            Document(content="doc1", id="1"),
            Document(content="doc2", id="2"),
        ]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
            RankerResult(document=Document(content="doc3", id="3"), score=0.7),
        ]

        recall = CompressionEvaluator.calculate_recall_at_k(
            ranked_results, relevant_docs, k=3
        )
        assert recall == 1.0

    def test_recall_at_k_partial_found(self) -> None:
        """Test recall with partial matches."""
        relevant_docs = [
            Document(content="doc1", id="1"),
            Document(content="doc2", id="2"),
            Document(content="doc3", id="3"),
            Document(content="doc4", id="4"),
        ]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
        ]

        recall = CompressionEvaluator.calculate_recall_at_k(
            ranked_results, relevant_docs, k=5
        )
        # 2 out of 4 found
        assert recall == 0.5

    def test_recall_at_k_none_found(self) -> None:
        """Test recall when no relevant docs found."""
        relevant_docs = [Document(content="docX", id="X")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
        ]

        recall = CompressionEvaluator.calculate_recall_at_k(
            ranked_results, relevant_docs, k=5
        )
        assert recall == 0.0

    def test_recall_at_k_respects_k(self) -> None:
        """Test that recall respects the k parameter."""
        relevant_docs = [Document(content="doc3", id="3")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
            RankerResult(document=Document(content="doc3", id="3"), score=0.7),
        ]

        # With k=2, doc3 is not in top-k
        recall_k2 = CompressionEvaluator.calculate_recall_at_k(
            ranked_results, relevant_docs, k=2
        )
        assert recall_k2 == 0.0

        # With k=3, doc3 is in top-k
        recall_k3 = CompressionEvaluator.calculate_recall_at_k(
            ranked_results, relevant_docs, k=3
        )
        assert recall_k3 == 1.0

    def test_recall_at_k_empty_relevant(self) -> None:
        """Test recall with empty relevant docs."""
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
        ]

        recall = CompressionEvaluator.calculate_recall_at_k(ranked_results, [], k=5)
        assert recall == 0.0

    def test_recall_at_k_empty_ranked(self) -> None:
        """Test recall with empty ranked results."""
        relevant_docs = [Document(content="doc1", id="1")]

        recall = CompressionEvaluator.calculate_recall_at_k([], relevant_docs, k=5)
        assert recall == 0.0

    def test_recall_at_k_with_none_ids(self) -> None:
        """Test recall handling documents with None IDs."""
        relevant_docs = [Document(content="doc1", id="1")]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id=None), score=0.8),
        ]

        recall = CompressionEvaluator.calculate_recall_at_k(
            ranked_results, relevant_docs, k=2
        )
        assert recall == 1.0


class TestCompressionEvaluatorEvaluateCompression:
    """Tests for CompressionEvaluator.evaluate_compression method."""

    def test_evaluate_compression_full(self) -> None:
        """Test comprehensive evaluation with all metrics."""
        ideal_docs = [
            Document(content="doc1", id="1"),
            Document(content="doc2", id="2"),
        ]

        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
            RankerResult(document=Document(content="doc2", id="2"), score=0.8),
        ]

        metrics = CompressionEvaluator.evaluate_compression(
            ranked_results=ranked_results,
            ideal_results=ideal_docs,
            original_token_count=1000,
            compressed_token_count=500,
            k=10,
        )

        assert isinstance(metrics, CompressionEvaluationMetrics)
        assert metrics.ndcg == 1.0
        assert metrics.mrr == 1.0
        assert 1 in metrics.recall_at_k
        assert 5 in metrics.recall_at_k
        assert 10 in metrics.recall_at_k
        assert 20 in metrics.recall_at_k
        assert metrics.tokens_saved == 500
        assert metrics.compression_ratio == 0.5
        assert metrics.mean_score == pytest.approx(0.85, abs=0.001)

    def test_evaluate_compression_no_savings(self) -> None:
        """Test evaluation when compressed is larger."""
        ideal_docs = [Document(content="doc1", id="1")]
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
        ]

        metrics = CompressionEvaluator.evaluate_compression(
            ranked_results=ranked_results,
            ideal_results=ideal_docs,
            original_token_count=500,
            compressed_token_count=1000,
            k=10,
        )

        # Tokens saved should be 0 (not negative)
        assert metrics.tokens_saved == 0
        # Compression ratio > 1 means expansion, not compression
        assert metrics.compression_ratio == 2.0

    def test_evaluate_compression_zero_original(self) -> None:
        """Test evaluation with zero original tokens."""
        ideal_docs = [Document(content="doc1", id="1")]
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
        ]

        metrics = CompressionEvaluator.evaluate_compression(
            ranked_results=ranked_results,
            ideal_results=ideal_docs,
            original_token_count=0,
            compressed_token_count=0,
            k=10,
        )

        assert metrics.compression_ratio == 1.0
        assert metrics.tokens_saved == 0

    def test_evaluate_compression_empty_results(self) -> None:
        """Test evaluation with empty results."""
        ideal_docs = [Document(content="doc1", id="1")]

        metrics = CompressionEvaluator.evaluate_compression(
            ranked_results=[],
            ideal_results=ideal_docs,
            original_token_count=1000,
            compressed_token_count=500,
            k=10,
        )

        assert metrics.ndcg == 0.0
        assert metrics.mrr == 0.0
        assert metrics.mean_score == 0.0

    def test_evaluate_compression_recall_values(self) -> None:
        """Test that recall is calculated for multiple k values."""
        ideal_docs = [Document(content="doc1", id="1")]
        ranked_results = [
            RankerResult(document=Document(content="doc1", id="1"), score=0.9),
        ]

        metrics = CompressionEvaluator.evaluate_compression(
            ranked_results=ranked_results,
            ideal_results=ideal_docs,
            original_token_count=100,
            compressed_token_count=50,
            k=10,
        )

        # All recall values should be 1.0 since we found the doc
        for k_val in [1, 5, 10, 20]:
            assert metrics.recall_at_k[k_val] == 1.0


class TestPrintDetailedReport:
    """Tests for print_detailed_report function."""

    def test_print_detailed_report_output(self, capsys) -> None:
        """Test that report prints all expected sections."""
        metrics = CompressionEvaluationMetrics(
            ndcg=0.85,
            mrr=0.90,
            recall_at_k={1: 0.7, 5: 0.85, 10: 0.90, 20: 0.95},
            tokens_saved=1000,
            compression_ratio=0.5,
            mean_score=0.88,
        )

        print_detailed_report(metrics, pipeline_name="Test Pipeline")

        captured = capsys.readouterr()
        output = captured.out

        # Check for expected sections
        assert "Test Pipeline - Detailed Evaluation Report" in output
        assert "Ranking Quality Metrics:" in output
        assert "NDCG@10:" in output
        assert "0.8500" in output
        assert "MRR:" in output
        assert "0.9000" in output
        assert "Recall Metrics:" in output
        assert "Recall@ 1:" in output
        assert "Recall@ 5:" in output
        assert "Recall@10:" in output
        assert "Recall@20:" in output
        assert "Compression Metrics:" in output
        assert "Compression Ratio:" in output
        assert "0.5000" in output
        assert "Tokens Saved:" in output
        assert "1000" in output

    def test_print_detailed_report_default_name(self, capsys) -> None:
        """Test report with default pipeline name."""
        metrics = CompressionEvaluationMetrics(
            ndcg=0.5,
            mrr=0.5,
            recall_at_k={1: 0.5},
            tokens_saved=100,
            compression_ratio=0.8,
            mean_score=0.5,
        )

        print_detailed_report(metrics)

        captured = capsys.readouterr()
        assert "Compression Pipeline - Detailed Evaluation Report" in captured.out

    def test_print_detailed_report_zero_metrics(self, capsys) -> None:
        """Test report with zero metrics."""
        metrics = CompressionEvaluationMetrics(
            ndcg=0.0,
            mrr=0.0,
            recall_at_k={1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0},
            tokens_saved=0,
            compression_ratio=1.0,
            mean_score=0.0,
        )

        print_detailed_report(metrics, pipeline_name="Empty Pipeline")

        captured = capsys.readouterr()
        assert "Empty Pipeline - Detailed Evaluation Report" in captured.out
        assert "0.0000" in captured.out
