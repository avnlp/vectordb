"""Dataset loaders for evaluation and indexing."""

from vectordb.dataloaders.base import BaseDatasetLoader
from vectordb.dataloaders.catalog import DataloaderCatalog
from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.datasets.arc import ARCLoader
from vectordb.dataloaders.datasets.earnings_calls import EarningsCallsLoader
from vectordb.dataloaders.datasets.factscore import FactScoreLoader
from vectordb.dataloaders.datasets.popqa import PopQALoader
from vectordb.dataloaders.datasets.triviaqa import TriviaQALoader
from vectordb.dataloaders.types import (
    DataloaderError,
    DatasetLoadError,
    DatasetRecord,
    DatasetType,
    DatasetValidationError,
    EvaluationQuery,
    UnsupportedDatasetError,
)


__all__ = [
    "ARCLoader",
    "BaseDatasetLoader",
    "DataloaderCatalog",
    "DataloaderError",
    "DatasetLoadError",
    "DatasetRecord",
    "DatasetType",
    "DatasetValidationError",
    "EarningsCallsLoader",
    "EvaluationQuery",
    "FactScoreLoader",
    "LoadedDataset",
    "PopQALoader",
    "TriviaQALoader",
    "UnsupportedDatasetError",
]
