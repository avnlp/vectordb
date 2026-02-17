"""Base dataloader interface for dataset-specific implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.types import (
    DatasetLoadError,
    DatasetRecord,
    DatasetType,
    DatasetValidationError,
)


class BaseDatasetLoader(ABC):
    """Define the dataset loading contract."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        limit: int | None = None,
        streaming: bool = True,
    ) -> None:
        """Initialize the loader with dataset configuration.

        Args:
            dataset_name: HuggingFace dataset identifier.
            split: Dataset split to load.
            limit: Optional limit on the number of records to emit.
            streaming: Whether to stream the dataset when possible.

        Raises:
            DatasetValidationError: If split is empty or limit is negative.
        """
        if not split:
            raise DatasetValidationError("Split must be a non-empty string.")
        if limit is not None and limit < 0:
            raise DatasetValidationError("Limit must be zero or a positive integer.")

        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self.streaming = streaming

    @property
    @abstractmethod
    def dataset_type(self) -> DatasetType:
        """Return the supported dataset type identifier."""

    @abstractmethod
    def _load_dataset_iterable(self) -> Iterable[Mapping[str, Any]]:
        """Return the raw dataset rows as an iterable."""

    @abstractmethod
    def _parse_row(self, row: Mapping[str, Any]) -> list[DatasetRecord]:
        """Parse a dataset row into normalized records."""

    def load(self) -> LoadedDataset:
        """Load the dataset and return normalized records.

        Returns:
            LoadedDataset containing normalized records.

        Raises:
            DatasetLoadError: If loading fails unexpectedly.
        """
        try:
            dataset_iterable = self._load_dataset_iterable()
        except Exception as exc:  # pragma: no cover - handled in tests via mocking
            raise DatasetLoadError("Failed to load dataset.") from exc

        records: list[DatasetRecord] = []

        try:
            for row in dataset_iterable:
                for record in self._parse_row(row):
                    records.append(record)
                    # Limit is record-based because TriviaQA expands rows into
                    # multiple evidence documents.
                    if self.limit is not None and len(records) >= self.limit:
                        return LoadedDataset(self.dataset_type, records)
        except DatasetValidationError:
            raise
        except Exception as exc:
            raise DatasetLoadError("Failed to parse dataset rows.") from exc

        return LoadedDataset(self.dataset_type, records)
