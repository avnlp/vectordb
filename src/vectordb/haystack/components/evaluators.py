"""DeepEval integration for RAG evaluation metrics.

Provides utilities to evaluate RAG pipelines using contextual metrics:
- Contextual Relevancy: How relevant are retrieved documents to the query?
- Contextual Recall: What percentage of expected information is in retrieved docs?
- Answer Relevancy: How relevant is the generated answer to the query?
- Faithfulness: Is the answer grounded in the retrieved documents?

DeepEval is an optional dependency; the evaluator gracefully handles its absence
by raising a descriptive error at initialization time rather than at import time.

Usage:
    >>> from vectordb.haystack.components import DeepEvalEvaluator
    >>> evaluator = DeepEvalEvaluator()
    >>> results = evaluator.evaluate_all(
    ...     query="What is RAG?",
    ...     retrieval_context=["RAG is retrieval-augmented generation..."],
    ...     answer="RAG combines retrieval with generation...",
    ...     expected_output="RAG stands for retrieval-augmented generation",
    ... )

Note:
    Install with: pip install deepeval
"""

import logging
from typing import Any, Optional


logger = logging.getLogger(__name__)


class DeepEvalEvaluator:
    """Wrapper for DeepEval metrics.

    Provides consistent interface for evaluating RAG pipeline outputs.

    DeepEval uses LLM-as-a-judge to evaluate RAG quality. Each metric
    returns a score (0.0-1.0) and optional reasoning for the score.

    Metrics:
        - contextual_recall: Measures coverage of expected output in retrieved docs
        - contextual_precision: Measures signal-to-noise in retrieved docs
        - answer_relevancy: Measures how well answer addresses the query
        - faithfulness: Measures hallucination (answer grounded in context)

    Attributes:
        ContextualRecallMetric: DeepEval metric class for recall.
        ContextualPrecisionMetric: DeepEval metric class for precision.
        AnswerRelevancyMetric: DeepEval metric class for answer relevance.
        FaithfulnessMetric: DeepEval metric class for faithfulness.

    Note:
        Metrics are imported lazily at initialization to avoid hard dependency.
    """

    def __init__(self) -> None:
        """Initialize DeepEval evaluator."""
        try:
            # Import here to avoid hard dependency at module load time
            # This allows the module to be imported even without deepeval installed
            # pylint: disable=import-outside-toplevel
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )

            # Store metric classes as instance attributes for later use
            self.ContextualRecallMetric: Any = ContextualRecallMetric
            self.ContextualPrecisionMetric: Any = ContextualPrecisionMetric
            self.AnswerRelevancyMetric: Any = AnswerRelevancyMetric
            self.FaithfulnessMetric: Any = FaithfulnessMetric
            logger.info("Initialized DeepEval evaluator")
        except ImportError as e:
            logger.error("DeepEval not installed: %s", str(e))
            raise ValueError(
                "DeepEval is required for evaluation. "
                "Install with: pip install deepeval"
            ) from e

    def evaluate_contextual_recall(
        self,
        retrieval_context: list[str],
        expected_output: str,
    ) -> dict[str, Any]:
        """Evaluate contextual recall.

        Args:
            retrieval_context: List of retrieved document contents.
            expected_output: Expected/ground truth information.

        Returns:
            Evaluation results including score and reasoning.
        """
        try:
            metric = self.ContextualRecallMetric(
                retrieval_context=retrieval_context,
                expected_output=expected_output,
            )
            metric.measure()
            return {
                "metric": "contextual_recall",
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            logger.error("Contextual recall evaluation failed: %s", str(e))
            return {
                "metric": "contextual_recall",
                "score": 0.0,
                "error": str(e),
            }

    def evaluate_contextual_precision(
        self,
        retrieval_context: list[str],
        expected_output: str,
    ) -> dict[str, Any]:
        """Evaluate contextual precision.

        Args:
            retrieval_context: List of retrieved document contents.
            expected_output: Expected/ground truth information.

        Returns:
            Evaluation results including score and reasoning.
        """
        try:
            metric = self.ContextualPrecisionMetric(
                retrieval_context=retrieval_context,
                expected_output=expected_output,
            )
            metric.measure()
            return {
                "metric": "contextual_precision",
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            logger.error("Contextual precision evaluation failed: %s", str(e))
            return {
                "metric": "contextual_precision",
                "score": 0.0,
                "error": str(e),
            }

    def evaluate_answer_relevancy(
        self,
        input_text: str,
        actual_output: str,
    ) -> dict[str, Any]:
        """Evaluate answer relevancy.

        Args:
            input_text: The query/question.
            actual_output: The generated answer.

        Returns:
            Evaluation results including score and reasoning.
        """
        try:
            metric = self.AnswerRelevancyMetric(
                input=input_text,
                actual_output=actual_output,
            )
            metric.measure()
            return {
                "metric": "answer_relevancy",
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            logger.error("Answer relevancy evaluation failed: %s", str(e))
            return {
                "metric": "answer_relevancy",
                "score": 0.0,
                "error": str(e),
            }

    def evaluate_faithfulness(
        self,
        retrieval_context: list[str],
        actual_output: str,
    ) -> dict[str, Any]:
        """Evaluate faithfulness (grounding in context).

        Args:
            retrieval_context: List of retrieved document contents.
            actual_output: The generated answer.

        Returns:
            Evaluation results including score and reasoning.
        """
        try:
            metric = self.FaithfulnessMetric(
                retrieval_context=retrieval_context,
                actual_output=actual_output,
            )
            metric.measure()
            return {
                "metric": "faithfulness",
                "score": metric.score,
                "reason": getattr(metric, "reason", ""),
            }
        except Exception as e:
            logger.error("Faithfulness evaluation failed: %s", str(e))
            return {
                "metric": "faithfulness",
                "score": 0.0,
                "error": str(e),
            }

    def evaluate_all(
        self,
        query: str,
        retrieval_context: list[str],
        answer: str,
        expected_output: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run all evaluation metrics.

        Args:
            query: The query/question.
            retrieval_context: List of retrieved document contents.
            answer: The generated answer.
            expected_output: Optional ground truth (for recall evaluation).

        Returns:
            Dictionary with all metric scores.
        """
        results = {
            "answer_relevancy": self.evaluate_answer_relevancy(query, answer),
            "faithfulness": self.evaluate_faithfulness(retrieval_context, answer),
        }

        if expected_output:
            results["contextual_recall"] = self.evaluate_contextual_recall(
                retrieval_context, expected_output
            )
            results["contextual_precision"] = self.evaluate_contextual_precision(
                retrieval_context, expected_output
            )

        return results
