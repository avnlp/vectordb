"""Earnings calls dataset loader."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from datasets import load_dataset as hf_load_dataset

from vectordb.dataloaders.base import BaseDatasetLoader
from vectordb.dataloaders.types import (
    DatasetRecord,
    DatasetType,
    DatasetValidationError,
)


class EarningsCallsLoader(BaseDatasetLoader):
    """Load earnings calls dataset rows into normalized records."""

    def __init__(
        self,
        dataset_name: str = "lamini/earnings-calls-qa",
        split: str = "train",
        limit: int | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the earnings calls loader.

        Args:
            dataset_name: HuggingFace dataset identifier.
            split: Dataset split to load.
            limit: Optional limit on record count.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            dataset_name=dataset_name, split=split, limit=limit, streaming=streaming
        )

    @property
    def dataset_type(self) -> DatasetType:
        """Return the dataset identifier."""
        return "earnings_calls"

    def _load_dataset_iterable(self) -> Iterable[Mapping[str, Any]]:
        """Load raw earnings calls dataset rows."""
        return hf_load_dataset(
            self.dataset_name, split=self.split, streaming=self.streaming
        )

    def _parse_row(self, row: Mapping[str, Any]) -> list[DatasetRecord]:
        """Parse a single earnings calls row into a record.

        Converts raw earnings call transcript data into a normalized record
        with the extracted question-answer pair and comprehensive metadata.
        Attempts to parse the quarter identifier into year and quarter
        components, but gracefully preserves the raw value for malformed
        data to avoid data loss.
        """
        try:
            question = row["question"]
            answer = row["answer"]
            transcript = row["transcript"]
            ticker = row["ticker"]
        except KeyError as exc:
            raise DatasetValidationError(
                "Missing required earnings calls fields."
            ) from exc

        year: int | None = None
        quarter: str | None = None
        raw_quarter = row.get("q")
        # Parse the quarter identifier (format: "YYYY-Qx") when available.
        # We attempt structured parsing but preserve the raw value even if
        # malformed, since discarding data may be problematic for evaluation.
        if isinstance(raw_quarter, str) and "-" in raw_quarter:
            year_str, quarter_str = raw_quarter.split("-", maxsplit=1)
            if year_str.isdigit():
                year = int(year_str)
                quarter = quarter_str

        metadata: dict[str, Any] = {
            "question": question,
            "answers": [str(answer)],
            "ticker": ticker,
            "company": row.get("company") or ticker,
            "date": row.get("date"),
            "year": year,
            "quarter": quarter,
            "raw_quarter": raw_quarter,
        }
        if row.get("id") is not None:
            metadata["id"] = row.get("id")

        return [DatasetRecord(text=transcript, metadata=metadata)]
