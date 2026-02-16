"""Evaluation data extraction utilities.

This module provides tools for extracting query-answer pairs from datasets
in a format suitable for RAG evaluation pipelines. The EvaluationExtractor
handles the variations in how different datasets store questions and answers,
providing a unified interface for evaluation.

Design Rationale:
    Different datasets use varying field names for questions ("question", "entity")
    and answers ("answers", "answer", "possible_answers"). This module abstracts
    these differences to provide consistent evaluation data.

Supported Dataset Formats:
    - TriviaQA: Uses "question" and "answers" (list of aliases)
    - ARC: Uses "question" and "answer" (single correct choice)
    - PopQA: Uses "question" and "possible_answers"
    - FactScore: Uses "entity" as question proxy and facts as answers
    - Earnings Calls: Uses "question" and "answer"

Deduplication Strategy:
    The extractor maintains a seen_questions set to avoid duplicate queries
    in the evaluation set. This is important because some datasets may have
    multiple documents per question, and we want unique queries for evaluation.

Usage Patterns:
    Direct extraction for evaluation:
        >>> from vectordb.dataloaders import EvaluationExtractor
        >>> queries = EvaluationExtractor.extract("triviaqa", split="test", limit=100)
        >>> for q in queries:
        ...     print(q["query"], q["answers"])

    Integration with evaluation frameworks:
        >>> queries = EvaluationExtractor.extract("popqa", limit=50)
        >>> for query_data in queries:
        ...     response = rag_pipeline.run(query_data["query"])
        ...     score = evaluate_answer(response, query_data["answers"])
"""

from __future__ import annotations

from typing import Any

from vectordb.dataloaders.loaders import DatasetRegistry


class EvaluationExtractor:
    """Extracts query-answer pairs for evaluation pipelines.

    This utility class provides a standardized way to extract evaluation data
    from various datasets. It handles field name variations and answer format
    differences across datasets.

    The extractor performs deduplication to ensure each query appears only
    once in the evaluation set, even if the dataset contains multiple
    documents per question.
    """

    @classmethod
    def extract(
        cls,
        dataset_type: str,
        dataset_name: str | None = None,
        split: str = "test",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Extract queries and ground truth answers from a dataset.

        Loads the specified dataset and extracts query-answer pairs suitable
        for RAG evaluation. Handles variations in field naming across
        different datasets.

        Args:
            dataset_type: Dataset type identifier (e.g., "triviaqa", "arc",
                "popqa", "factscore", "earnings_calls")
            dataset_name: HuggingFace dataset ID (optional, uses defaults
                if not provided)
            split: Dataset split to load (default: "test")
            limit: Maximum number of queries to return (None for all)

        Returns:
            List of dicts with structure:
            {
                "query": str,  # The question text
                "answers": list[str],  # Ground truth answers (always a list)
                "metadata": dict  # Additional fields excluding query/answer
            }

        Raises:
            ValueError: If dataset_type is not supported by DatasetRegistry

        Example:
            >>> queries = EvaluationExtractor.extract("triviaqa", limit=10)
            >>> queries[0].keys()
            dict_keys(['query', 'answers', 'metadata'])
        """
        data = DatasetRegistry.load(
            dataset_type,  # type: ignore[arg-type]
            dataset_name=dataset_name,
            split=split,
        )

        # Track seen questions to avoid duplicates in evaluation set
        # Some datasets have multiple documents per question
        seen_questions: set[str] = set()
        result: list[dict[str, Any]] = []

        for item in data:
            meta = item["metadata"]

            # Extract question text - different datasets use different field names
            # Priority: "question" field, then "entity" (used by FactScore)
            question = meta.get("question") or meta.get("entity")
            if question is None or question in seen_questions:
                # Skip if no question found or already processed
                continue

            seen_questions.add(question)

            # Extract answers - normalize to list format
            # Priority: "answers" (list), then "answer" (single value)
            answers = meta.get("answers") or meta.get("answer")
            if answers is None:
                answers = []
            # Normalize single answers to list for consistent evaluation interface
            if isinstance(answers, str):
                answers = [answers]

            # Build result with query, answers, and filtered metadata
            result.append(
                {
                    "query": question,
                    "answers": answers,
                    "metadata": {
                        # Include all metadata except query/answer fields
                        # to avoid redundancy while preserving context
                        k: v
                        for k, v in meta.items()
                        if k not in ("question", "answers", "entity", "answer")
                    },
                }
            )

            # Early termination when limit is reached
            if limit is not None and len(result) >= limit:
                break

        return result
