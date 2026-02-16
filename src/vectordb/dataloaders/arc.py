"""ARC (AI2 Reasoning Challenge) dataset loader.

This module provides a dataloader for the ARC dataset, which consists of
science questions requiring reasoning to answer. The dataset is designed
to test advanced question answering and reasoning capabilities.

Dataset Structure:
    - question: Science question requiring reasoning
    - choices: Multiple choice options with labels (A, B, C, D...)
    - answerKey: The correct answer label
    - id: Unique identifier for the question

Use Case:
    Useful for evaluating RAG systems on science QA tasks where
    the answer requires understanding scientific concepts and reasoning.

Data Source:
    HuggingFace dataset: ai2_arc (configs: ARC-Challenge, ARC-Easy)
    The Challenge subset contains more difficult questions requiring
    multi-step reasoning, while Easy contains more straightforward queries.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset as hf_load_dataset


logger = logging.getLogger(__name__)


class ARCDataloader:
    """Loader for ARC (AI2 Reasoning Challenge) dataset.

    This dataloader handles the ARC dataset format, which presents science
    questions as multiple-choice problems. The loader formats questions
    with their choices to create indexable documents that preserve the
    full context needed for QA tasks.

    The standardized output format enables consistent processing across
    different RAG evaluation frameworks.

    Attributes:
        dataset_name: HuggingFace dataset identifier (default: "ai2_arc")
        config: Dataset configuration - "ARC-Challenge" or "ARC-Easy"
        split: Dataset split to load (default: "validation")
        limit: Maximum number of items to load (None for all)
    """

    def __init__(
        self,
        dataset_name: str = "ai2_arc",
        config: str = "ARC-Challenge",
        split: str = "validation",
        limit: int | None = None,
    ) -> None:
        """Initialize ARC dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            config: Dataset configuration (ARC-Challenge or ARC-Easy)
            split: Dataset split to load
            limit: Maximum number of items to load (None for all)
        """
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.limit = limit

    def load(self) -> list[dict[str, Any]]:
        """Load ARC dataset and return standardized format.

        This method streams through the dataset and formats each question
        with its multiple-choice options for indexing. The formatted
        question includes all choices to provide context for the QA task.

        The streaming approach (streaming=True) is used to handle large
        datasets efficiently without loading everything into memory.

        Returns:
            List of standardized dicts with 'text' (formatted question)
            and 'metadata' (original data including answer key).

            Each dict has structure:
            {
                "text": "Question text\nChoices: A) ... B) ...",
                "metadata": {
                    "question": "Original question",
                    "choices": {"label": [...], "text": [...]},
                    "answer_key": "A",
                    "id": "unique_id"
                }
            }
        """
        dataset = hf_load_dataset(
            self.dataset_name, self.config, split=self.split, streaming=True
        )
        result: list[dict[str, Any]] = []

        for row in dataset:
            question = row["question"]
            choices = row["choices"]
            answer_key = row["answerKey"]

            formatted_question = f"{question}\nChoices: "
            for label, text in zip(choices["label"], choices["text"]):
                formatted_question += f"{label}) {text} "

            result.append(
                {
                    "text": formatted_question,
                    "metadata": {
                        "question": question,
                        "choices": choices,
                        "answer_key": answer_key,
                        "id": row["id"],
                    },
                }
            )
            # Early termination when limit is reached
            if self.limit and len(result) >= self.limit:
                break

        logger.info("Loaded %d items from ARC", len(result))
        return result
