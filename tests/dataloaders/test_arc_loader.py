"""Unit tests for ARC loader."""

from unittest.mock import patch

from vectordb.dataloaders.base import BaseDatasetLoader
from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.datasets.arc import ARCLoader
from vectordb.dataloaders.types import DatasetValidationError


class TestARCLoaderInitialization:
    """Tests for ARC loader initialization."""

    def test_defaults(self) -> None:
        """Test that ARCLoader initializes with correct default values."""
        loader = ARCLoader()

        assert loader.dataset_name == "ai2_arc"
        assert loader.config == "ARC-Challenge"
        assert loader.split == "validation"
        assert loader.limit is None
        assert loader.streaming is True

    def test_custom_values(self) -> None:
        """Test that ARCLoader accepts and stores custom initialization values."""
        loader = ARCLoader(
            dataset_name="custom_arc",
            config="ARC-Easy",
            split="test",
            limit=10,
            streaming=False,
        )

        assert loader.dataset_name == "custom_arc"
        assert loader.config == "ARC-Easy"
        assert loader.split == "test"
        assert loader.limit == 10
        assert loader.streaming is False

    def test_rejects_empty_split(self) -> None:
        """Test that ARCLoader raises DatasetValidationError for empty split."""
        try:
            ARCLoader(split="")
        except DatasetValidationError:
            assert True
        else:
            raise AssertionError

    def test_rejects_negative_limit(self) -> None:
        """Test that ARCLoader raises DatasetValidationError for negative limit."""
        try:
            ARCLoader(limit=-1)
        except DatasetValidationError:
            assert True
        else:
            raise AssertionError


class TestARCLoaderLoad:
    """Tests for ARC loader load behavior."""

    def test_load_returns_loaded_dataset(
        self, arc_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that load() returns a LoadedDataset instance."""
        loader = ARCLoader()

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(arc_sample_rows)
            dataset = loader.load()

        assert isinstance(dataset, LoadedDataset)

    def test_load_records_shape(self, arc_sample_rows, make_streaming_dataset) -> None:
        """Test that loaded records have correct structure with text and metadata."""
        loader = ARCLoader()

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(arc_sample_rows[:1])
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.text
        assert "question" in record.metadata
        assert "answers" in record.metadata

    def test_load_formats_choices(
        self, arc_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that answer choices are formatted in the record text."""
        loader = ARCLoader()

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(arc_sample_rows[:1])
            dataset = loader.load()

        record = dataset.records()[0]
        assert "Choices:" in record.text
        assert "A)" in record.text

    def test_load_maps_answer_key(
        self, arc_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that answer key is correctly mapped to the answer text in metadata."""
        loader = ARCLoader()

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(arc_sample_rows[:1])
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.metadata["answers"] == ["Paris"]

    def test_load_respects_limit(self, arc_sample_rows, make_streaming_dataset) -> None:
        """Test that load() respects the limit parameter."""
        loader = ARCLoader(limit=1)

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(arc_sample_rows)
            dataset = loader.load()

        assert len(dataset.records()) == 1

    def test_load_empty_dataset(self, make_streaming_dataset) -> None:
        """Test that load() handles empty datasets correctly."""
        loader = ARCLoader()

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset([])
            dataset = loader.load()

        assert dataset.records() == []

    def test_load_streaming_enabled(
        self, arc_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that load() passes streaming=True to the dataset loader."""
        loader = ARCLoader()

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(arc_sample_rows[:1])
            loader.load()

        mock_load.assert_called_once_with(
            "ai2_arc",
            "ARC-Challenge",
            split="validation",
            streaming=True,
        )


class TestARCLoaderEdgeCases:
    """Tests for ARC loader edge cases."""

    def test_load_invalid_answer_key(
        self, arc_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that load() raises DatasetValidationError for invalid answer keys."""
        loader = ARCLoader()
        broken_rows = [dict(arc_sample_rows[0], answerKey="Z")]

        with patch("vectordb.dataloaders.datasets.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = make_streaming_dataset(broken_rows)
            try:
                loader.load()
            except DatasetValidationError:
                assert True
            else:
                raise AssertionError

    def test_loader_is_base_subclass(self) -> None:
        """Test that ARCLoader is a subclass of BaseDatasetLoader."""
        assert issubclass(ARCLoader, BaseDatasetLoader)
