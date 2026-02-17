"""Unit tests for earnings calls loader."""

from unittest.mock import patch

from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.datasets.earnings_calls import EarningsCallsLoader


class TestEarningsCallsLoaderInitialization:
    """Tests for earnings calls loader initialization."""

    def test_defaults(self) -> None:
        """Test that loader initializes with correct default values."""
        loader = EarningsCallsLoader()

        assert loader.dataset_name == "lamini/earnings-calls-qa"
        assert loader.split == "train"
        assert loader.limit is None
        assert loader.streaming is True

    def test_custom_values(self) -> None:
        """Test that loader initializes with custom values."""
        loader = EarningsCallsLoader(dataset_name="custom", split="test", limit=4)

        assert loader.dataset_name == "custom"
        assert loader.split == "test"
        assert loader.limit == 4


class TestEarningsCallsLoaderLoad:
    """Tests for earnings calls loader load behavior."""

    def test_load_returns_loaded_dataset(
        self, earnings_calls_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that load() returns a LoadedDataset instance."""
        loader = EarningsCallsLoader()

        with patch(
            "vectordb.dataloaders.datasets.earnings_calls.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(earnings_calls_sample_rows)
            dataset = loader.load()

        assert isinstance(dataset, LoadedDataset)

    def test_transcript_mapping(
        self, earnings_calls_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that transcript field maps correctly to record text."""
        loader = EarningsCallsLoader()

        with patch(
            "vectordb.dataloaders.datasets.earnings_calls.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(earnings_calls_sample_rows)
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.text == "Q4 earnings call transcript for Acme Corp."

    def test_quarter_parsing(
        self, earnings_calls_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that quarter string is parsed into year and quarter metadata."""
        loader = EarningsCallsLoader()

        with patch(
            "vectordb.dataloaders.datasets.earnings_calls.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(earnings_calls_sample_rows)
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["year"] == 2023
        assert record.metadata["quarter"] == "Q4"

    def test_malformed_quarter(
        self, earnings_calls_edge_bad_q, make_streaming_dataset
    ) -> None:
        """Test that malformed quarter strings are handled gracefully."""
        loader = EarningsCallsLoader()

        with patch(
            "vectordb.dataloaders.datasets.earnings_calls.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(earnings_calls_edge_bad_q)
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["year"] is None
        assert record.metadata["quarter"] is None
        assert record.metadata["raw_quarter"] == "not-a-quarter"

    def test_company_fallback(
        self, earnings_calls_edge_bad_q, make_streaming_dataset
    ) -> None:
        """Test that company field falls back to ticker when name is missing."""
        loader = EarningsCallsLoader()

        with patch(
            "vectordb.dataloaders.datasets.earnings_calls.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(earnings_calls_edge_bad_q)
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["company"] == "ACME"


class TestEarningsCallsLoaderEdgeCases:
    """Tests for earnings calls loader edge cases."""

    def test_load_streaming_enabled(
        self, earnings_calls_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that streaming is enabled when loading the dataset."""
        loader = EarningsCallsLoader()

        with patch(
            "vectordb.dataloaders.datasets.earnings_calls.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(earnings_calls_sample_rows)
            loader.load()

        mock_load.assert_called_once_with(
            "lamini/earnings-calls-qa",
            split="train",
            streaming=True,
        )
