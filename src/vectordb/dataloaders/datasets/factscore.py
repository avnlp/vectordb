"""FactScore dataset loader."""

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


class FactScoreLoader(BaseDatasetLoader):
    """Load FactScore dataset rows into normalized records."""

    def __init__(
        self,
        dataset_name: str = "dskar/FActScore",
        split: str = "test",
        limit: int | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the FactScore loader.

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
        return "factscore"

    def _load_dataset_iterable(self) -> Iterable[Mapping[str, Any]]:
        """Load raw FactScore dataset rows."""
        return hf_load_dataset(
            self.dataset_name, split=self.split, streaming=self.streaming
        )

    def _parse_row(self, row: Mapping[str, Any]) -> list[DatasetRecord]:
        """Parse a single FactScore row into a record.

        Extracts Wikipedia content and fact claims for a given entity.
        Normalizes fact representations which may be stored as either plain
        strings or nested dictionaries with a "fact" key, ensuring consistent
        downstream processing. Prioritizes decomposed facts when available
        for more granular evaluation.
        """
        try:
            entity = row["entity"]
            wikipedia_text = row["wikipedia_text"]
        except KeyError as exc:
            raise DatasetValidationError("Missing required FactScore fields.") from exc

        decomposed_facts = row.get("decomposed_facts", [])
        facts = row.get("facts", [])
        # Prefer decomposed facts for evaluation when available, as they provide
        # more granular factual claims compared to the aggregated facts list.
        answers_source = decomposed_facts if decomposed_facts else facts
        normalized_answers: list[str] = []
        # Facts may be stored in two formats: plain strings or dictionaries
        # with a "fact" key. This unified handling accommodates both formats.
        for value in answers_source:
            if isinstance(value, str) and value.strip():
                normalized_answers.append(value.strip())
            elif isinstance(value, dict) and "fact" in value:
                fact_value = value.get("fact")
                if isinstance(fact_value, str) and fact_value.strip():
                    normalized_answers.append(fact_value.strip())

        metadata: dict[str, Any] = {
            "question": entity,
            "answers": normalized_answers,
            "entity": entity,
            "topic": row.get("topic", entity),
            "facts": facts,
            "decomposed_facts": decomposed_facts,
            "id": row.get("id"),
            "one_fact_prompt": row.get("one_fact_prompt"),
            "factscore_prompt": row.get("factscore_prompt"),
        }

        return [DatasetRecord(text=wikipedia_text, metadata=metadata)]
