"""ARC dataset loader."""

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


class ARCLoader(BaseDatasetLoader):
    """Load ARC dataset rows into normalized records."""

    def __init__(
        self,
        dataset_name: str = "ai2_arc",
        config: str = "ARC-Challenge",
        split: str = "validation",
        limit: int | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the ARC loader.

        Args:
            dataset_name: HuggingFace dataset identifier.
            config: Dataset configuration name.
            split: Dataset split to load.
            limit: Optional limit on record count.
            streaming: Whether to stream the dataset.
        """
        super().__init__(
            dataset_name=dataset_name, split=split, limit=limit, streaming=streaming
        )
        self.config = config

    @property
    def dataset_type(self) -> DatasetType:
        """Return the dataset identifier."""
        return "arc"

    def _load_dataset_iterable(self) -> Iterable[Mapping[str, Any]]:
        """Load raw ARC dataset rows."""
        return hf_load_dataset(
            self.dataset_name,
            self.config,
            split=self.split,
            streaming=self.streaming,
        )

    def _parse_row(self, row: Mapping[str, Any]) -> list[DatasetRecord]:
        """Parse a single ARC row into a record.

        Extracts the question and answer choices from the raw ARC format,
        formatting them as a complete question string with all options
        included. The correct answer is identified by matching the
        answerKey to the choice labels.
        """
        try:
            question = row["question"]
            choices = row["choices"]
            labels = choices["label"]
            texts = choices["text"]
            answer_key = row["answerKey"]
        except KeyError as exc:
            raise DatasetValidationError("Missing required ARC fields.") from exc

        if answer_key not in labels:
            raise DatasetValidationError("ARC answerKey missing from labels.")

        # Format the question with all available answer choices to provide
        # comprehensive context for retrieval and evaluation.
        formatted_question = f"{question}\nChoices: "
        formatted_question += " ".join(
            f"{label}) {text}" for label, text in zip(labels, texts)
        )
        # Find the correct answer's text by locating its corresponding index
        # in the labels list.
        answer_index = labels.index(answer_key)

        metadata: dict[str, Any] = {
            "question": question,
            "answers": [texts[answer_index]],
            "answer_key": answer_key,
            "choices": choices,
            "id": row.get("id"),
        }

        return [DatasetRecord(text=formatted_question, metadata=metadata)]
