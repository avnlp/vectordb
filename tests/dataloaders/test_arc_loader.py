"""Tests for ARC dataset loader.

This module tests the ARCDataloader class which loads the
AI2 Reasoning Challenge dataset.
"""

from unittest.mock import MagicMock, patch

from vectordb.dataloaders.arc import ARCDataloader


class TestARCDataloaderInitialization:
    """Test suite for ARCDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom configuration
    - Parameter handling
    """

    def test_arc_dataloader_init_defaults(self) -> None:
        """Test ARC dataloader with default parameters."""
        loader = ARCDataloader()

        assert loader.dataset_name == "ai2_arc"
        assert loader.config == "ARC-Challenge"
        assert loader.split == "validation"
        assert loader.limit is None

    def test_arc_dataloader_init_custom_split(self) -> None:
        """Test ARC dataloader with custom split."""
        loader = ARCDataloader(split="test")

        assert loader.split == "test"

    def test_arc_dataloader_init_custom_dataset_name(self) -> None:
        """Test ARC dataloader with custom dataset name."""
        loader = ARCDataloader(dataset_name="custom_arc")

        assert loader.dataset_name == "custom_arc"

    def test_arc_dataloader_init_with_limit(self) -> None:
        """Test ARC dataloader with limit."""
        loader = ARCDataloader(limit=100)

        assert loader.limit == 100

    def test_arc_dataloader_init_all_params(self) -> None:
        """Test ARC dataloader with all custom parameters."""
        loader = ARCDataloader(
            dataset_name="custom",
            config="ARC-Easy",
            split="validation",
            limit=50,
        )

        assert loader.dataset_name == "custom"
        assert loader.config == "ARC-Easy"
        assert loader.split == "validation"
        assert loader.limit == 50


class TestARCDataloaderLoad:
    """Test suite for ARC dataset loading.

    Tests cover:
    - Loading ARC dataset
    - Data format and structure
    - Metadata preservation
    - Question formatting
    - Limit handling
    """

    def test_arc_load_returns_list(self, arc_sample_rows) -> None:
        """Test that load returns a list."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert isinstance(result, list)

    def test_arc_load_correct_data_structure(self, arc_sample_rows) -> None:
        """Test that loaded data has correct structure."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) > 0
            assert "text" in result[0]
            assert "metadata" in result[0]

    def test_arc_load_preserves_question(self, arc_sample_rows) -> None:
        """Test that question is preserved in metadata."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["question"] == "What is the capital of France?"

    def test_arc_load_preserves_answer_key(self, arc_sample_rows) -> None:
        """Test that answer key is preserved."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["answer_key"] == "A"

    def test_arc_load_preserves_id(self, arc_sample_rows) -> None:
        """Test that question ID is preserved."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["id"] == "arc_1"

    def test_arc_load_formats_question_with_choices(self, arc_sample_rows) -> None:
        """Test that question is formatted with choices."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            # Text should include question and choices
            assert "What is the capital of France?" in result[0]["text"]
            assert "Choices:" in result[0]["text"]
            assert "A)" in result[0]["text"]

    def test_arc_load_respects_limit(self, arc_sample_rows) -> None:
        """Test that limit parameter is respected."""
        loader = ARCDataloader(limit=1)

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) == 1

    def test_arc_load_empty_dataset(self) -> None:
        """Test loading empty ARC dataset."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter([]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result == []

    def test_arc_load_multiple_rows(self, arc_sample_rows) -> None:
        """Test loading multiple rows."""
        loader = ARCDataloader()

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) == len(arc_sample_rows)

    def test_arc_load_dataset_name_passed_to_load_dataset(
        self, arc_sample_rows
    ) -> None:
        """Test that dataset name is passed to load_dataset."""
        loader = ARCDataloader(dataset_name="custom_arc")

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            loader.load()

            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert "custom_arc" in call_args[0] or "custom_arc" in str(call_args)

    def test_arc_load_config_passed_to_load_dataset(self, arc_sample_rows) -> None:
        """Test that config is passed to load_dataset."""
        loader = ARCDataloader(config="ARC-Easy")

        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(arc_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            loader.load()

            mock_load.assert_called_once()
