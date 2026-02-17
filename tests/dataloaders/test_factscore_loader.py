"""Unit tests for FactScore loader."""

from unittest.mock import patch

from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.datasets.factscore import FactScoreLoader


class TestFactScoreLoaderInitialization:
    """Tests for FactScore loader initialization."""

    def test_defaults(self) -> None:
        """Test that FactScoreLoader initializes with correct default values."""
        loader = FactScoreLoader()

        assert loader.dataset_name == "dskar/FActScore"
        assert loader.split == "test"
        assert loader.limit is None
        assert loader.streaming is True

    def test_custom_values(self) -> None:
        """Test that FactScoreLoader accepts and stores custom initialization values."""
        loader = FactScoreLoader(dataset_name="custom", split="validation", limit=1)

        assert loader.dataset_name == "custom"
        assert loader.split == "validation"
        assert loader.limit == 1


class TestFactScoreLoaderLoad:
    """Tests for FactScore loader load behavior."""

    def test_load_returns_loaded_dataset(
        self, factscore_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that load() returns a LoadedDataset instance."""
        loader = FactScoreLoader()

        with patch(
            "vectordb.dataloaders.datasets.factscore.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(factscore_sample_rows)
            dataset = loader.load()

        assert isinstance(dataset, LoadedDataset)

    def test_wikipedia_text_mapping(
        self, factscore_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that Wikipedia text is correctly mapped to record text field."""
        loader = FactScoreLoader()

        with patch(
            "vectordb.dataloaders.datasets.factscore.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(factscore_sample_rows)
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.text == "Albert Einstein was a German-born physicist."

    def test_topic_entity_mapping(
        self, factscore_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that topic and entity are correctly mapped to metadata fields."""
        loader = FactScoreLoader()

        with patch(
            "vectordb.dataloaders.datasets.factscore.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(factscore_sample_rows)
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["question"] == "Albert Einstein"
        assert record.metadata["entity"] == "Albert Einstein"

    def test_answers_fallback(
        self, factscore_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that answers metadata field is populated from available data."""
        loader = FactScoreLoader()

        with patch(
            "vectordb.dataloaders.datasets.factscore.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(factscore_sample_rows)
            dataset = loader.load()

        record = dataset.records()[0]
        assert "Einstein was born in Germany" in record.metadata["answers"]

    def test_defaults_for_optional(
        self, factscore_edge_missing_optional, make_streaming_dataset
    ) -> None:
        """Test optional fields default to safe values when missing."""
        loader = FactScoreLoader()

        with patch(
            "vectordb.dataloaders.datasets.factscore.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(
                factscore_edge_missing_optional
            )
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["topic"] == "Ada Lovelace"
        assert record.metadata["facts"] == []
        assert record.metadata["decomposed_facts"] == []


class TestFactScoreLoaderEdgeCases:
    """Tests for FactScore loader edge cases."""

    def test_load_streaming_enabled(
        self, factscore_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that streaming mode is enabled when loading the dataset."""
        loader = FactScoreLoader()

        with patch(
            "vectordb.dataloaders.datasets.factscore.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(factscore_sample_rows)
            loader.load()

        mock_load.assert_called_once_with(
            "dskar/FActScore",
            split="test",
            streaming=True,
        )
