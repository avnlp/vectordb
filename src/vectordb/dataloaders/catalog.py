"""Catalog for dataset loader creation."""

from __future__ import annotations

from typing import Literal

from vectordb.dataloaders.base import BaseDatasetLoader
from vectordb.dataloaders.datasets.arc import ARCLoader
from vectordb.dataloaders.datasets.earnings_calls import EarningsCallsLoader
from vectordb.dataloaders.datasets.factscore import FactScoreLoader
from vectordb.dataloaders.datasets.popqa import PopQALoader
from vectordb.dataloaders.datasets.triviaqa import TriviaQALoader
from vectordb.dataloaders.types import DatasetType, UnsupportedDatasetError


class DataloaderCatalog:
    """Create dataset loaders by name."""

    _REGISTRY: dict[DatasetType, type[BaseDatasetLoader]] = {
        "triviaqa": TriviaQALoader,
        "arc": ARCLoader,
        "popqa": PopQALoader,
        "factscore": FactScoreLoader,
        "earnings_calls": EarningsCallsLoader,
    }

    @classmethod
    def create(
        cls,
        name: Literal["triviaqa", "arc", "popqa", "factscore", "earnings_calls"],
        split: str = "test",
        limit: int | None = None,
        dataset_id: str | None = None,
    ) -> BaseDatasetLoader:
        """Create a dataset loader.

        Args:
            name: Dataset type name.
            split: Dataset split to load.
            limit: Optional limit on record count.
            dataset_id: Optional HuggingFace dataset override.

        Returns:
            A configured dataset loader.

        Raises:
            UnsupportedDatasetError: If name is not supported.
        """
        if name not in cls._REGISTRY:
            raise UnsupportedDatasetError(f"Unsupported dataset: {name}")

        loader_cls = cls._REGISTRY[name]
        kwargs = {"dataset_name": dataset_id} if dataset_id is not None else {}
        return loader_cls(split=split, limit=limit, **kwargs)

    @classmethod
    def supported_datasets(cls) -> tuple[DatasetType, ...]:
        """Return supported dataset identifiers."""
        return tuple(cls._REGISTRY.keys())
