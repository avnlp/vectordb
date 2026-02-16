"""Tests for PopQA dataset loader.

This module tests the PopQADataloader class which loads the
PopQA factoid question answering dataset.
"""

from unittest.mock import MagicMock, patch

from vectordb.dataloaders.popqa import PopQADataloader


class TestPopQADataloaderInitialization:
    """Test suite for PopQADataloader initialization.

    Tests cover:
    - Default configuration
    - Custom configuration
    - Parameter handling
    """

    def test_popqa_dataloader_init_defaults(self) -> None:
        """Test PopQA dataloader with default parameters."""
        loader = PopQADataloader()

        assert loader.dataset_name == "akariasai/PopQA"
        assert loader.split == "test"

    def test_popqa_dataloader_init_custom_split(self) -> None:
        """Test PopQA dataloader with custom split."""
        loader = PopQADataloader(split="validation")

        assert loader.split == "validation"

    def test_popqa_dataloader_init_custom_dataset_name(self) -> None:
        """Test PopQA dataloader with custom dataset name."""
        loader = PopQADataloader(dataset_name="custom_popqa")

        assert loader.dataset_name == "custom_popqa"

    def test_popqa_dataloader_init_with_limit(self) -> None:
        """Test PopQA dataloader with limit."""
        loader = PopQADataloader(limit=50)

        assert loader.limit == 50

    def test_popqa_dataloader_init_all_params(self) -> None:
        """Test PopQA dataloader with all custom parameters."""
        loader = PopQADataloader(
            dataset_name="custom",
            split="validation",
            limit=50,
        )

        assert loader.dataset_name == "custom"
        assert loader.split == "validation"
        assert loader.limit == 50


class TestPopQADataloaderLoad:
    """Test suite for PopQA dataset loading.

    Tests cover:
    - Loading PopQA dataset
    - Data format and structure
    - Metadata preservation
    - Question and entity handling
    - Limit handling
    """

    def test_popqa_load_returns_list(self, popqa_sample_rows) -> None:
        """Test that load returns a list."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert isinstance(result, list)

    def test_popqa_load_correct_data_structure(self, popqa_sample_rows) -> None:
        """Test that loaded data has correct structure."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) > 0
            assert "text" in result[0]
            assert "metadata" in result[0]

    def test_popqa_load_preserves_question(self, popqa_sample_rows) -> None:
        """Test that question is preserved in metadata."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["question"] == "What is the capital of France?"

    def test_popqa_load_preserves_entity(self, popqa_sample_rows) -> None:
        """Test that entity is preserved in metadata."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["entity"] == "France"

    def test_popqa_load_preserves_answers(self, popqa_sample_rows) -> None:
        """Test that answers are preserved."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert "answers" in result[0]["metadata"]
            assert "Paris" in result[0]["metadata"]["answers"]

    def test_popqa_load_includes_content(self, popqa_sample_rows) -> None:
        """Test that content is included in text field."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert (
                "Paris is the capital and largest city of France." in result[0]["text"]
            )

    def test_popqa_load_respects_limit(self, popqa_sample_rows) -> None:
        """Test that limit parameter is respected."""
        loader = PopQADataloader(limit=1)

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) == 1

    def test_popqa_load_empty_dataset(self) -> None:
        """Test loading empty PopQA dataset."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter([]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result == []

    def test_popqa_load_multiple_rows(self, popqa_sample_rows) -> None:
        """Test loading multiple rows."""
        loader = PopQADataloader()

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) == len(popqa_sample_rows)

    def test_popqa_load_dataset_name_passed(self, popqa_sample_rows) -> None:
        """Test that dataset name is passed to load_dataset."""
        loader = PopQADataloader(dataset_name="custom_popqa")

        with patch("vectordb.dataloaders.popqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(popqa_sample_rows[:1]))
            mock_load.return_value = mock_dataset

            loader.load()

            mock_load.assert_called_once()
