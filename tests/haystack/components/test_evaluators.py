"""Comprehensive tests for Haystack evaluator components.

This module tests the DeepEvalEvaluator which provides RAG evaluation metrics
including contextual recall, precision, answer relevancy, and faithfulness.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestDeepEvalEvaluator:
    """Test suite for DeepEvalEvaluator component.

    Tests cover:
    - Evaluator initialization (with and without deepeval)
    - Contextual recall evaluation
    - Contextual precision evaluation
    - Answer relevancy evaluation
    - Faithfulness evaluation
    - Combined evaluation with evaluate_all
    - Error handling for each metric
    - Edge cases
    """

    @pytest.fixture
    def mock_deepeval_metrics(self):
        """Fixture for mocked DeepEval metrics."""
        # Patch the deepeval module directly since it's imported inside __init__
        with (
            patch("deepeval.metrics.ContextualRecallMetric") as mock_recall,
            patch("deepeval.metrics.ContextualPrecisionMetric") as mock_precision,
            patch("deepeval.metrics.AnswerRelevancyMetric") as mock_relevancy,
            patch("deepeval.metrics.FaithfulnessMetric") as mock_faithfulness,
        ):
            # Setup mock metric instances
            mock_recall_instance = MagicMock()
            mock_recall_instance.score = 0.85
            mock_recall_instance.reason = "Good recall of relevant information"
            mock_recall.return_value = mock_recall_instance

            mock_precision_instance = MagicMock()
            mock_precision_instance.score = 0.90
            mock_precision_instance.reason = "High precision in retrieval"
            mock_precision.return_value = mock_precision_instance

            mock_relevancy_instance = MagicMock()
            mock_relevancy_instance.score = 0.88
            mock_relevancy_instance.reason = "Answer is relevant to query"
            mock_relevancy.return_value = mock_relevancy_instance

            mock_faithfulness_instance = MagicMock()
            mock_faithfulness_instance.score = 0.92
            mock_faithfulness_instance.reason = "Answer is faithful to context"
            mock_faithfulness.return_value = mock_faithfulness_instance

            yield {
                "recall": (mock_recall, mock_recall_instance),
                "precision": (mock_precision, mock_precision_instance),
                "relevancy": (mock_relevancy, mock_relevancy_instance),
                "faithfulness": (mock_faithfulness, mock_faithfulness_instance),
            }

    def test_initialization_success(self, mock_deepeval_metrics):
        """Test successful DeepEvalEvaluator initialization."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()

        assert evaluator is not None
        assert evaluator.ContextualRecallMetric is not None
        assert evaluator.ContextualPrecisionMetric is not None
        assert evaluator.AnswerRelevancyMetric is not None
        assert evaluator.FaithfulnessMetric is not None

    def test_evaluate_contextual_recall_success(self, mock_deepeval_metrics):
        """Test successful contextual recall evaluation."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        retrieval_context = ["Document 1 content", "Document 2 content"]
        expected_output = "Expected answer based on documents"

        result = evaluator.evaluate_contextual_recall(
            retrieval_context, expected_output
        )

        assert result["metric"] == "contextual_recall"
        assert result["score"] == 0.85
        assert result["reason"] == "Good recall of relevant information"

        # Verify metric was called correctly
        mock_cls, mock_instance = mock_deepeval_metrics["recall"]
        mock_cls.assert_called_once_with(
            retrieval_context=retrieval_context,
            expected_output=expected_output,
        )
        mock_instance.measure.assert_called_once()

    def test_evaluate_contextual_recall_exception(self, mock_deepeval_metrics):
        """Test contextual recall evaluation with exception."""
        mock_cls, mock_instance = mock_deepeval_metrics["recall"]
        mock_instance.measure.side_effect = Exception("Metric calculation failed")

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_contextual_recall(["doc"], "expected")

        assert result["metric"] == "contextual_recall"
        assert result["score"] == 0.0
        assert "error" in result
        assert "Metric calculation failed" in result["error"]

    def test_evaluate_contextual_precision_success(self, mock_deepeval_metrics):
        """Test successful contextual precision evaluation."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        retrieval_context = ["Document 1", "Document 2"]
        expected_output = "Expected output"

        result = evaluator.evaluate_contextual_precision(
            retrieval_context, expected_output
        )

        assert result["metric"] == "contextual_precision"
        assert result["score"] == 0.90
        assert result["reason"] == "High precision in retrieval"

        mock_cls, mock_instance = mock_deepeval_metrics["precision"]
        mock_cls.assert_called_once_with(
            retrieval_context=retrieval_context,
            expected_output=expected_output,
        )

    def test_evaluate_contextual_precision_exception(self, mock_deepeval_metrics):
        """Test contextual precision evaluation with exception."""
        mock_cls, mock_instance = mock_deepeval_metrics["precision"]
        mock_instance.measure.side_effect = Exception("Precision calculation error")

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_contextual_precision(["doc"], "expected")

        assert result["metric"] == "contextual_precision"
        assert result["score"] == 0.0
        assert "error" in result

    def test_evaluate_answer_relevancy_success(self, mock_deepeval_metrics):
        """Test successful answer relevancy evaluation."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        query = "What is machine learning?"
        answer = "Machine learning is a subset of AI."

        result = evaluator.evaluate_answer_relevancy(query, answer)

        assert result["metric"] == "answer_relevancy"
        assert result["score"] == 0.88
        assert result["reason"] == "Answer is relevant to query"

        mock_cls, mock_instance = mock_deepeval_metrics["relevancy"]
        mock_cls.assert_called_once_with(
            input=query,
            actual_output=answer,
        )

    def test_evaluate_answer_relevancy_exception(self, mock_deepeval_metrics):
        """Test answer relevancy evaluation with exception."""
        mock_cls, mock_instance = mock_deepeval_metrics["relevancy"]
        mock_instance.measure.side_effect = Exception("Relevancy calculation error")

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_answer_relevancy("query", "answer")

        assert result["metric"] == "answer_relevancy"
        assert result["score"] == 0.0
        assert "error" in result

    def test_evaluate_faithfulness_success(self, mock_deepeval_metrics):
        """Test successful faithfulness evaluation."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        retrieval_context = ["Context document 1", "Context document 2"]
        answer = "Answer based on the context."

        result = evaluator.evaluate_faithfulness(retrieval_context, answer)

        assert result["metric"] == "faithfulness"
        assert result["score"] == 0.92
        assert result["reason"] == "Answer is faithful to context"

        mock_cls, mock_instance = mock_deepeval_metrics["faithfulness"]
        mock_cls.assert_called_once_with(
            retrieval_context=retrieval_context,
            actual_output=answer,
        )

    def test_evaluate_faithfulness_exception(self, mock_deepeval_metrics):
        """Test faithfulness evaluation with exception."""
        mock_cls, mock_instance = mock_deepeval_metrics["faithfulness"]
        mock_instance.measure.side_effect = Exception("Faithfulness calculation error")

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_faithfulness(["doc"], "answer")

        assert result["metric"] == "faithfulness"
        assert result["score"] == 0.0
        assert "error" in result

    def test_evaluate_all_with_expected_output(self, mock_deepeval_metrics):
        """Test evaluate_all with all metrics including expected output."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        query = "What is AI?"
        retrieval_context = ["AI is artificial intelligence."]
        answer = "AI refers to machine intelligence."
        expected_output = "AI is the simulation of human intelligence."

        result = evaluator.evaluate_all(
            query=query,
            retrieval_context=retrieval_context,
            answer=answer,
            expected_output=expected_output,
        )

        assert "answer_relevancy" in result
        assert "faithfulness" in result
        assert "contextual_recall" in result
        assert "contextual_precision" in result

        assert result["answer_relevancy"]["score"] == 0.88
        assert result["faithfulness"]["score"] == 0.92
        assert result["contextual_recall"]["score"] == 0.85
        assert result["contextual_precision"]["score"] == 0.90

    def test_evaluate_all_without_expected_output(self, mock_deepeval_metrics):
        """Test evaluate_all without expected output.

        Only relevancy and faithfulness metrics.
        """
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        query = "What is AI?"
        retrieval_context = ["AI is artificial intelligence."]
        answer = "AI refers to machine intelligence."

        result = evaluator.evaluate_all(
            query=query,
            retrieval_context=retrieval_context,
            answer=answer,
        )

        assert "answer_relevancy" in result
        assert "faithfulness" in result
        assert "contextual_recall" not in result
        assert "contextual_precision" not in result

    def test_evaluate_all_empty_context(self, mock_deepeval_metrics):
        """Test evaluate_all with empty retrieval context."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()

        result = evaluator.evaluate_all(
            query="What is AI?",
            retrieval_context=[],
            answer="AI is artificial intelligence.",
        )

        assert "answer_relevancy" in result
        assert "faithfulness" in result

    def test_metric_no_reason_attribute(self, mock_deepeval_metrics):
        """Test handling of metrics without reason attribute."""
        mock_cls, mock_instance = mock_deepeval_metrics["recall"]
        # Remove reason attribute
        del mock_instance.reason

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_contextual_recall(["doc"], "expected")

        assert result["metric"] == "contextual_recall"
        assert result["score"] == 0.85
        assert result["reason"] == ""  # Should default to empty string

    def test_metric_reason_is_none(self, mock_deepeval_metrics):
        """Test handling of metrics with None reason."""
        mock_cls, mock_instance = mock_deepeval_metrics["precision"]
        mock_instance.reason = None

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_contextual_precision(["doc"], "expected")

        # getattr returns None when reason is None
        assert result["reason"] is None

    def test_evaluate_contextual_recall_empty_context(self, mock_deepeval_metrics):
        """Test contextual recall with empty context."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_contextual_recall([], "expected")

        assert result["metric"] == "contextual_recall"
        # Should still work even with empty context

    def test_evaluate_faithfulness_empty_context(self, mock_deepeval_metrics):
        """Test faithfulness with empty context."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_faithfulness([], "answer")

        assert result["metric"] == "faithfulness"

    def test_evaluate_answer_relevancy_empty_strings(self, mock_deepeval_metrics):
        """Test answer relevancy with empty strings."""
        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_answer_relevancy("", "")

        assert result["metric"] == "answer_relevancy"

    def test_all_metrics_with_zero_scores(self, mock_deepeval_metrics):
        """Test all metrics returning zero scores."""
        for _metric_name, (_mock_cls, mock_instance) in mock_deepeval_metrics.items():
            mock_instance.score = 0.0
            mock_instance.reason = "Very poor performance"

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_all(
            query="query",
            retrieval_context=["doc"],
            answer="answer",
            expected_output="expected",
        )

        assert result["contextual_recall"]["score"] == 0.0
        assert result["contextual_precision"]["score"] == 0.0
        assert result["answer_relevancy"]["score"] == 0.0
        assert result["faithfulness"]["score"] == 0.0

    def test_all_metrics_with_perfect_scores(self, mock_deepeval_metrics):
        """Test all metrics returning perfect scores."""
        for _metric_name, (_mock_cls, mock_instance) in mock_deepeval_metrics.items():
            mock_instance.score = 1.0
            mock_instance.reason = "Perfect performance"

        from vectordb.haystack.components.evaluators import DeepEvalEvaluator

        evaluator = DeepEvalEvaluator()
        result = evaluator.evaluate_all(
            query="query",
            retrieval_context=["doc"],
            answer="answer",
            expected_output="expected",
        )

        assert result["contextual_recall"]["score"] == 1.0
        assert result["contextual_precision"]["score"] == 1.0
        assert result["answer_relevancy"]["score"] == 1.0
        assert result["faithfulness"]["score"] == 1.0
