"""Unit tests for PopQA loader."""

from unittest.mock import patch

from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.datasets.popqa import PopQALoader


class TestPopQALoaderInitialization:
    """Tests for PopQA loader initialization."""

    def test_defaults(self) -> None:
        """Test that PopQALoader uses correct default values."""
        loader = PopQALoader()

        assert loader.dataset_name == "akariasai/PopQA"
        assert loader.split == "test"
        assert loader.limit is None
        assert loader.streaming is True

    def test_custom_values(self) -> None:
        """Test that PopQALoader correctly accepts and stores custom values."""
        loader = PopQALoader(dataset_name="custom", split="train", limit=2)

        assert loader.dataset_name == "custom"
        assert loader.split == "train"
        assert loader.limit == 2


class TestPopQALoaderLoad:
    """Tests for PopQA loader load behavior."""

    def test_load_returns_loaded_dataset(
        self, popqa_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that load() returns a LoadedDataset instance."""
        loader = PopQALoader()

        with patch("vectordb.dataloaders.datasets.popqa.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(popqa_sample_rows)
            dataset = loader.load()

        assert isinstance(dataset, LoadedDataset)

    def test_metadata_mapping(self, popqa_sample_rows, make_streaming_dataset) -> None:
        """Test that entity, predicate, and object are correctly mapped to metadata."""
        loader = PopQALoader()

        with patch("vectordb.dataloaders.datasets.popqa.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(popqa_sample_rows[:1])
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["entity"] == "France"
        assert record.metadata["predicate"] == "capital"
        assert record.metadata["object"] == "Paris"

    def test_content_fallback(
        self, popqa_edge_missing_content, make_streaming_dataset
    ) -> None:
        """Test text falls back to question when content field is missing."""
        loader = PopQALoader()

        with patch("vectordb.dataloaders.datasets.popqa.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(popqa_edge_missing_content)
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.text == "What is the capital of France?"

    def test_answers_normalized(
        self, popqa_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that answers are normalized from pipe-separated string to list."""
        loader = PopQALoader()

        with patch("vectordb.dataloaders.datasets.popqa.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(popqa_sample_rows[:1])
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["answers"] == ["Paris"]

    def test_limit_and_empty(self, popqa_sample_rows, make_streaming_dataset) -> None:
        """Test limit restricts record count and empty datasets return empty."""
        loader = PopQALoader(limit=1)

        with patch("vectordb.dataloaders.datasets.popqa.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(popqa_sample_rows)
            dataset = loader.load()

        assert len(dataset.records()) == 1

        with patch("vectordb.dataloaders.datasets.popqa.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset([])
            dataset = loader.load()

        assert dataset.records() == []


class TestPopQALoaderEdgeCases:
    """Tests for PopQA loader edge cases."""

    def test_load_streaming_enabled(
        self, popqa_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that streaming mode is enabled by default when loading the dataset."""
        loader = PopQALoader()

        with patch("vectordb.dataloaders.datasets.popqa.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(popqa_sample_rows[:1])
            loader.load()

        mock_load.assert_called_once_with(
            "akariasai/PopQA",
            split="test",
            streaming=True,
        )
