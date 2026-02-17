"""PopQA dataset loader."""

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


class PopQALoader(BaseDatasetLoader):
    """Load PopQA dataset rows into normalized records."""

    def __init__(
        self,
        dataset_name: str = "akariasai/PopQA",
        split: str = "test",
        limit: int | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the PopQA loader.

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
        return "popqa"

    def _load_dataset_iterable(self) -> Iterable[Mapping[str, Any]]:
        """Load raw PopQA dataset rows."""
        return hf_load_dataset(
            self.dataset_name, split=self.split, streaming=self.streaming
        )

    def _parse_row(self, row: Mapping[str, Any]) -> list[DatasetRecord]:
        """Parse a single PopQA row into a record.

        Converts raw PopQA row data into a normalized record with the
        complete question context and properly normalized answer strings.
        Includes semantic triple information (subject, predicate, object)
        in the metadata for downstream evaluation.
        """
        try:
            question = row["question"]
            answers = row["possible_answers"]
            entity = row["subj"]
            predicate = row["prop"]
            obj = row["obj"]
        except KeyError as exc:
            raise DatasetValidationError("Missing required PopQA fields.") from exc

        # Use provided content if available, otherwise fall back to the question
        # to ensure we always have relevant text for retrieval.
        content = row.get("content") or question
        # Extract non-empty answer strings after whitespace normalization to
        # ensure clean evaluation metrics.
        normalized_answers = [
            value.strip()
            for value in answers
            if isinstance(value, str) and value.strip()
        ]

        metadata: dict[str, Any] = {
            "question": question,
            "answers": normalized_answers,
            "entity": entity,
            "predicate": predicate,
            "object": obj,
        }
        if row.get("id") is not None:
            metadata["id"] = row.get("id")

        return [DatasetRecord(text=content, metadata=metadata)]
