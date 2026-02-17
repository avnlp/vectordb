"""Unit tests for dataloader catalog."""

import pytest

from vectordb.dataloaders.catalog import DataloaderCatalog
from vectordb.dataloaders.datasets.arc import ARCLoader
from vectordb.dataloaders.datasets.earnings_calls import EarningsCallsLoader
from vectordb.dataloaders.datasets.factscore import FactScoreLoader
from vectordb.dataloaders.datasets.popqa import PopQALoader
from vectordb.dataloaders.datasets.triviaqa import TriviaQALoader
from vectordb.dataloaders.types import UnsupportedDatasetError


class TestDataloaderCatalog:
    """Tests for DataloaderCatalog behavior."""

    def test_create_resolves_classes(self) -> None:
        """Verify create method returns correct loader instances."""
        assert isinstance(DataloaderCatalog.create("triviaqa"), TriviaQALoader)
        assert isinstance(DataloaderCatalog.create("arc"), ARCLoader)
        assert isinstance(DataloaderCatalog.create("popqa"), PopQALoader)
        assert isinstance(DataloaderCatalog.create("factscore"), FactScoreLoader)
        assert isinstance(
            DataloaderCatalog.create("earnings_calls"), EarningsCallsLoader
        )

    def test_unknown_dataset_raises(self) -> None:
        """Verify unknown dataset raises UnsupportedDatasetError."""
        with pytest.raises(UnsupportedDatasetError):
            DataloaderCatalog.create("unknown")

    def test_dataset_id_override(self) -> None:
        """Verify dataset_id parameter overrides the loader's dataset name."""
        loader = DataloaderCatalog.create("arc", dataset_id="custom")

        assert loader.dataset_name == "custom"

    def test_supported_datasets_order(self) -> None:
        """Verify that supported_datasets returns datasets in expected order."""
        assert DataloaderCatalog.supported_datasets() == (
            "triviaqa",
            "arc",
            "popqa",
            "factscore",
            "earnings_calls",
        )
