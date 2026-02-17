"""Loaded dataset wrapper for conversions and evaluation queries."""

from __future__ import annotations

from typing import Any

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangChainDocument

from vectordb.dataloaders.converters import DocumentConverter, records_to_items
from vectordb.dataloaders.evaluation import EvaluationExtractor
from vectordb.dataloaders.types import DatasetRecord, DatasetType, EvaluationQuery


class LoadedDataset:
    """Hold normalized records and provide post-load operations."""

    def __init__(self, dataset_type: DatasetType, records: list[DatasetRecord]) -> None:
        """Initialize a loaded dataset.

        Args:
            dataset_type: Identifier of the dataset.
            records: Normalized dataset records.
        """
        self._dataset_type = dataset_type
        self._records = records

    def records(self) -> list[DatasetRecord]:
        """Return normalized dataset records."""
        return list(self._records)

    def to_dict_items(self) -> list[dict[str, Any]]:
        """Convert normalized records to dictionary items."""
        return records_to_items(self._records)

    def to_haystack(self) -> list[HaystackDocument]:
        """Convert records to Haystack documents."""
        return DocumentConverter.to_haystack(self.to_dict_items())

    def to_langchain(self) -> list[LangChainDocument]:
        """Convert records to LangChain documents."""
        return DocumentConverter.to_langchain(self.to_dict_items())

    def evaluation_queries(self, limit: int | None = None) -> list[EvaluationQuery]:
        """Extract evaluation queries from records.

        Args:
            limit: Optional limit applied after deduplication.

        Returns:
            List of evaluation queries.
        """
        return EvaluationExtractor.extract(self._records, limit=limit)
