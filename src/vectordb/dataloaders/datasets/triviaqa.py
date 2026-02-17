"""TriviaQA dataset loader."""

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


class TriviaQALoader(BaseDatasetLoader):
    """Load TriviaQA dataset rows into normalized records."""

    def __init__(
        self,
        dataset_name: str = "trivia_qa",
        config: str = "rc",
        split: str = "test",
        limit: int | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the TriviaQA loader.

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
        return "triviaqa"

    def _load_dataset_iterable(self) -> Iterable[Mapping[str, Any]]:
        """Load the raw TriviaQA dataset rows."""
        return hf_load_dataset(
            self.dataset_name,
            self.config,
            split=self.split,
            streaming=self.streaming,
        )

    def _parse_row(self, row: Mapping[str, Any]) -> list[DatasetRecord]:
        """Parse a single TriviaQA row into records.

        Converts a single TriviaQA row containing a question and multiple
        search result evidence documents into separate records. Each search
        result becomes an individual record with the full question context,
        allowing the ranking and ordering of retrieved evidence to influence
        retrieval evaluation metrics.
        """
        try:
            question = row["question"]
            search_results = row["search_results"]
        except KeyError as exc:
            raise DatasetValidationError("Missing required TriviaQA fields.") from exc

        # Normalize answers to a consistent list format, filtering out empty
        # values to ensure clean evaluation.
        answers = row.get("answer")
        if isinstance(answers, list):
            normalized_answers = [
                value.strip()
                for value in answers
                if isinstance(value, str) and value.strip()
            ]
        elif isinstance(answers, str) and answers.strip():
            normalized_answers = [answers.strip()]
        else:
            normalized_answers = []

        titles = search_results.get("title", [])
        contexts = search_results.get("search_context", [])
        descriptions = search_results.get("description", [])
        ranks = search_results.get("rank", [])
        row_id = row.get("id")
        records: list[DatasetRecord] = []

        for index, title in enumerate(titles):
            text_content = ""
            # Prefer search_context when available for richer evidence text,
            # as it typically contains more substantial context than descriptions.
            # Fall back to description for incomplete results.
            if index < len(contexts) and contexts[index]:
                text_content = contexts[index]
            elif index < len(descriptions):
                text_content = descriptions[index]

            metadata: dict[str, Any] = {
                "question": question,
                "answers": normalized_answers,
                "title": title,
                "rank": ranks[index] if index < len(ranks) else None,
                "evidence_index": index,
            }
            if row_id is not None:
                metadata["id"] = row_id

            records.append(DatasetRecord(text=text_content, metadata=metadata))

        return records
