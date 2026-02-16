"""Earnings Call transcripts loader for financial Q&A.

This module provides a dataloader for the Earnings Calls dataset, which contains
Q&A pairs from earnings call transcripts. It's useful for evaluating RAG on
financial domain queries.

Dataset Structure:
    - question: Financial question about the earnings call
    - answer: Answer extracted from transcript
    - date: Date of the earnings call
    - transcript: Full or partial transcript text
    - q: Quarter identifier (e.g., "2023-Q4")
    - ticker: Stock ticker symbol
    - company: Company name

Use Case:
    Evaluating RAG systems on financial domain queries about earnings calls.
    The dataset tests domain-specific retrieval with temporal and entity
    constraints (e.g., "What was Apple's revenue in Q3 2023?").

Data Source:
    HuggingFace dataset: lamini/earnings-calls-qa
    Contains transcripts from S&P 500 companies with associated Q&A pairs.

Temporal Considerations:
    The dataset includes quarter and year metadata, enabling evaluation of
    time-sensitive queries. This is crucial for financial RAG where answers
    depend on specific reporting periods.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset as hf_load_dataset


logger = logging.getLogger(__name__)


class EarningsCallDataloader:
    """Loader for Earnings Call transcripts dataset.

    This dataloader provides access to Q&A pairs from earnings call
    transcripts, useful for evaluating RAG on financial domain queries.
    Each transcript segment is paired with a question and answer.

    The loader extracts temporal metadata (year, quarter) from the "q" field
    which uses format "YYYY-QN" (e.g., "2023-Q4"). This enables filtering
    and retrieval based on specific reporting periods.

    Attributes:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        limit: Maximum number of items to load
    """

    def __init__(
        self,
        dataset_name: str = "lamini/earnings-calls-qa",
        split: str = "train",
        limit: int | None = None,
    ) -> None:
        """Initialize Earnings Call dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (default: "train")
            limit: Maximum number of items to load (None for all)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

    def load(self) -> list[dict[str, Any]]:
        """Load Earnings Call dataset and return standardized format.

        Processes the earnings call data by extracting temporal information
        from the quarter field and structuring the transcript as the indexable
        text content. The Q&A pairs are preserved in metadata for evaluation.

        Returns:
            List of standardized dicts with structure:
            {
                "text": "Transcript content...",
                "metadata": {
                    "question": "Financial question",
                    "answer": "Answer from transcript",
                    "ticker": "AAPL",
                    "date": "2023-10-25",
                    "quarter": "Q4",
                    "year": 2023,
                    "company": "Apple Inc."
                }
            }
        """
        dataset = hf_load_dataset(self.dataset_name, split=self.split, streaming=True)
        result: list[dict[str, Any]] = []

        for row in dataset:
            year_str, quarter = row["q"].split("-")
            result.append(
                {
                    "text": row["transcript"],
                    "metadata": {
                        "question": row["question"],
                        "answer": row["answer"],
                        "ticker": row["ticker"],
                        "date": row["date"],
                        "quarter": quarter,
                        "year": int(year_str),
                        "company": row.get("company", row.get("ticker")),
                    },
                }
            )
            if self.limit and len(result) >= self.limit:
                break

        logger.info("Loaded %d items from EarningsCalls", len(result))
        return result
