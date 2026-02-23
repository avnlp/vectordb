"""Tests for RAG evaluation module (evaluator.py).

Covers:
- RAGEvaluator: evaluate_query, evaluate_batch, get_report
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig
from vectordb.haystack.cost_optimized_rag.evaluation.evaluator import (
    RAGEvaluator,
)


@pytest.fixture
def mock_rag_config() -> RAGConfig:
    """Create a mock RAGConfig for testing."""
    return RAGConfig(
        collection={"name": "test_collection", "description": "Test"},
        dataloader={"type": "triviaqa", "dataset_name": "trivia_qa", "split": "test"},
        search={"top_k": 5},
        logging={"name": "test_evaluator", "level": "DEBUG"},
    )


@pytest.fixture
def mock_searcher() -> MagicMock:
    """Create a mock searcher that returns predefined results."""
    searcher = MagicMock()
    searcher.search.return_value = [
        {"id": "doc1", "content": "Result 1", "score": 0.9},
        {"id": "doc2", "content": "Result 2", "score": 0.8},
        {"id": "doc3", "content": "Result 3", "score": 0.7},
    ]
    return searcher


@pytest.fixture
def sample_queries() -> list[dict[str, Any]]:
    """Create sample query batch for testing."""
    return [
        {
            "query_id": "q1",
            "query": "What is the capital of France?",
            "relevant_ids": ["doc1", "doc5"],
        },
        {
            "query_id": "q2",
            "query": "What is machine learning?",
            "relevant_ids": ["doc2", "doc3"],
        },
        {
            "query_id": "q3",
            "query": "Python programming language",
            "relevant_ids": ["doc4"],
        },
    ]


class TestRAGEvaluatorInit:
    """Tests for RAGEvaluator initialization."""

    def test_init_creates_evaluator(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluator initializes correctly with config and searcher."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)

        assert evaluator.config is mock_rag_config
        assert evaluator.searcher is mock_searcher
        assert evaluator.logger is not None
        assert evaluator.aggregator is not None

    def test_init_with_different_log_levels(self, mock_searcher: MagicMock) -> None:
        """Test evaluator with various logging levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = RAGConfig(
                collection={"name": "test", "description": "Test"},
                dataloader={"type": "triviaqa", "dataset_name": "t", "split": "test"},
                logging={"name": "test", "level": level},
            )
            evaluator = RAGEvaluator(config, mock_searcher)
            assert evaluator.logger is not None


class TestRAGEvaluatorEvaluateQuery:
    """Tests for RAGEvaluator.evaluate_query method."""

    def test_evaluate_query_returns_metrics(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_query returns expected metric keys."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_query(
            query_id="q1",
            query="test query",
            relevant_ids={"doc1", "doc2"},
            top_k=3,
        )

        # Check required keys
        assert "query_id" in result
        assert "query" in result
        assert "latency_ms" in result
        assert "retrieved_count" in result
        assert "relevant_count" in result
        assert "mrr" in result
        assert "recall_at_k" in result
        assert "precision_at_k" in result
        assert "ndcg_at_k" in result
        assert "map_at_k" in result

    def test_evaluate_query_values(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_query returns correct values."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_query(
            query_id="q1",
            query="test query",
            relevant_ids={"doc1"},
            top_k=3,
        )

        assert result["query_id"] == "q1"
        assert result["query"] == "test query"
        assert result["latency_ms"] > 0  # Should have some latency
        assert result["retrieved_count"] == 3
        assert result["relevant_count"] == 1
        assert result["mrr"] == 1.0  # doc1 is first

    def test_evaluate_query_uses_config_top_k_when_not_provided(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_query uses config top_k when not explicitly passed."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        evaluator.evaluate_query(
            query_id="q1",
            query="test query",
            relevant_ids={"doc1"},
        )

        # Verify searcher was called with config's top_k (5)
        mock_searcher.search.assert_called_once_with("test query", top_k=5)

    def test_evaluate_query_with_no_relevant_docs(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_query when no relevant docs in results."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_query(
            query_id="q1",
            query="test query",
            relevant_ids={"doc99", "doc100"},  # Not in results
            top_k=3,
        )

        assert result["mrr"] == 0.0
        assert result["recall_at_k"] == 0.0
        assert result["precision_at_k"] == 0.0

    def test_evaluate_query_with_empty_relevant_ids(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_query with empty relevant_ids set."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_query(
            query_id="q1",
            query="test query",
            relevant_ids=set(),
            top_k=3,
        )

        assert result["relevant_count"] == 0
        assert result["recall_at_k"] == 0.0

    def test_evaluate_query_adds_to_aggregator(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_query adds result to aggregator."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        evaluator.evaluate_query(
            query_id="q1",
            query="test query",
            relevant_ids={"doc1"},
            top_k=3,
        )

        assert len(evaluator.aggregator.results) == 1
        assert evaluator.aggregator.results[0]["query_id"] == "q1"

    def test_evaluate_query_handles_results_without_id(
        self, mock_rag_config: RAGConfig
    ) -> None:
        """Test evaluate_query handles results missing 'id' key."""
        searcher = MagicMock()
        searcher.search.return_value = [
            {"content": "Result 1"},  # No 'id' key
            {"id": "doc2", "content": "Result 2"},
        ]

        evaluator = RAGEvaluator(mock_rag_config, searcher)
        result = evaluator.evaluate_query(
            query_id="q1",
            query="test query",
            relevant_ids={"doc2"},
            top_k=2,
        )

        # Should handle missing id gracefully (empty string default)
        assert result["retrieved_count"] == 2


class TestRAGEvaluatorEvaluateBatch:
    """Tests for RAGEvaluator.evaluate_batch method."""

    def test_evaluate_batch_processes_all_queries(
        self,
        mock_rag_config: RAGConfig,
        mock_searcher: MagicMock,
        sample_queries: list[dict[str, Any]],
    ) -> None:
        """Test evaluate_batch processes all queries in batch."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_batch(sample_queries, top_k=3)

        assert "individual_results" in result
        assert len(result["individual_results"]) == 3
        assert result["total_queries"] == 3

    def test_evaluate_batch_returns_aggregate_metrics(
        self,
        mock_rag_config: RAGConfig,
        mock_searcher: MagicMock,
        sample_queries: list[dict[str, Any]],
    ) -> None:
        """Test evaluate_batch returns aggregated metrics."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_batch(sample_queries, top_k=3)

        assert "avg_mrr" in result
        assert "avg_recall_at_k" in result
        assert "avg_precision_at_k" in result
        assert "avg_ndcg_at_k" in result
        assert "avg_map_at_k" in result

    def test_evaluate_batch_with_empty_list(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_batch with empty query list."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_batch([], top_k=3)

        assert result["total_queries"] == 0
        assert result["individual_results"] == []

    def test_evaluate_batch_handles_missing_fields(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_batch handles queries with missing fields."""
        queries = [
            {"query": "test"},  # Missing query_id and relevant_ids
            {"query_id": "q2"},  # Missing query and relevant_ids
        ]

        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_batch(queries, top_k=3)

        # Should handle gracefully with defaults
        assert len(result["individual_results"]) == 2

    def test_evaluate_batch_uses_config_top_k(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluate_batch uses config top_k when not provided."""
        queries = [{"query_id": "q1", "query": "test", "relevant_ids": ["doc1"]}]

        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        evaluator.evaluate_batch(queries)

        # Should use config's top_k (5)
        mock_searcher.search.assert_called_with("test", top_k=5)


class TestRAGEvaluatorGetReport:
    """Tests for RAGEvaluator.get_report method."""

    def test_get_report_empty(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test get_report with no evaluations."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        report = evaluator.get_report()

        assert report["total_queries"] == 0

    def test_get_report_after_evaluations(
        self,
        mock_rag_config: RAGConfig,
        mock_searcher: MagicMock,
        sample_queries: list[dict[str, Any]],
    ) -> None:
        """Test get_report returns summary after evaluations."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        evaluator.evaluate_batch(sample_queries, top_k=3)
        report = evaluator.get_report()

        assert report["total_queries"] == 3
        assert "avg_mrr" in report
        assert "avg_recall_at_k" in report

    def test_get_report_after_single_query(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test get_report after single query evaluation."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        evaluator.evaluate_query("q1", "test", {"doc1"}, top_k=3)
        report = evaluator.get_report()

        assert report["total_queries"] == 1


class TestEvaluatorIntegration:
    """Integration tests combining evaluator components."""

    def test_full_evaluation_workflow(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test full evaluation workflow with multiple queries."""
        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)

        # Evaluate individual queries
        evaluator.evaluate_query("q1", "query 1", {"doc1"}, top_k=3)
        evaluator.evaluate_query("q2", "query 2", {"doc2", "doc3"}, top_k=3)

        # Get report
        report = evaluator.get_report()

        assert report["total_queries"] == 2
        assert "avg_mrr" in report
        assert report["avg_mrr"] > 0

    def test_batch_and_individual_produce_same_results(
        self,
        mock_rag_config: RAGConfig,
        mock_searcher: MagicMock,
        sample_queries: list[dict[str, Any]],
    ) -> None:
        """Test batch and individual evaluations produce consistent results."""
        # Fresh evaluator for batch
        batch_evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        batch_result = batch_evaluator.evaluate_batch(sample_queries, top_k=3)

        # Fresh evaluator for individual
        individual_evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        for q in sample_queries:
            individual_evaluator.evaluate_query(
                q["query_id"],
                q["query"],
                set(q["relevant_ids"]),
                top_k=3,
            )
        individual_result = individual_evaluator.get_report()

        # Should have same aggregated metrics
        assert batch_result["total_queries"] == individual_result["total_queries"]
        assert batch_result["avg_mrr"] == individual_result["avg_mrr"]


class TestEvaluatorEdgeCases:
    """Edge case and error condition tests."""

    def test_evaluate_with_slow_searcher(self, mock_rag_config: RAGConfig) -> None:
        """Test evaluation with slow searcher captures latency."""
        import time

        def slow_search(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            time.sleep(0.05)
            return [{"id": "doc1"}]

        slow_searcher = MagicMock()
        slow_searcher.search.side_effect = slow_search

        evaluator = RAGEvaluator(mock_rag_config, slow_searcher)
        result = evaluator.evaluate_query("q1", "test", {"doc1"}, top_k=1)

        # Should capture latency in ms (at least 50ms)
        assert result["latency_ms"] >= 50

    def test_evaluate_with_empty_search_results(
        self, mock_rag_config: RAGConfig
    ) -> None:
        """Test evaluation when searcher returns empty results."""
        empty_searcher = MagicMock()
        empty_searcher.search.return_value = []

        evaluator = RAGEvaluator(mock_rag_config, empty_searcher)
        result = evaluator.evaluate_query("q1", "test", {"doc1"}, top_k=3)

        assert result["retrieved_count"] == 0
        assert result["mrr"] == 0.0

    def test_large_batch_evaluation(
        self, mock_rag_config: RAGConfig, mock_searcher: MagicMock
    ) -> None:
        """Test evaluation with large batch of queries."""
        queries = [
            {"query_id": f"q{i}", "query": f"query {i}", "relevant_ids": ["doc1"]}
            for i in range(100)
        ]

        evaluator = RAGEvaluator(mock_rag_config, mock_searcher)
        result = evaluator.evaluate_batch(queries, top_k=3)

        assert result["total_queries"] == 100
        assert len(result["individual_results"]) == 100
