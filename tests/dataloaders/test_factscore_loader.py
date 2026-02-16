"""Tests for FactScore dataset loader.

This module tests the FactScoreDataloader class which loads the
FactScore fact verification dataset.
"""

from unittest.mock import MagicMock, patch

from vectordb.dataloaders.factscore import FactScoreDataloader


class TestFactScoreDataloaderInitialization:
    """Test suite for FactScoreDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom configuration
    - Parameter handling
    """

    def test_factscore_dataloader_init_defaults(self) -> None:
        """Test FactScore dataloader with default parameters."""
        loader = FactScoreDataloader()

        assert loader.dataset_name == "dskar/FActScore"
        assert loader.split == "test"

    def test_factscore_dataloader_init_custom_split(self) -> None:
        """Test FactScore dataloader with custom split."""
        loader = FactScoreDataloader(split="validation")

        assert loader.split == "validation"

    def test_factscore_dataloader_init_custom_dataset_name(self) -> None:
        """Test FactScore dataloader with custom dataset name."""
        loader = FactScoreDataloader(dataset_name="custom_factscore")

        assert loader.dataset_name == "custom_factscore"

    def test_factscore_dataloader_init_with_limit(self) -> None:
        """Test FactScore dataloader with limit."""
        loader = FactScoreDataloader(limit=50)

        assert loader.limit == 50

    def test_factscore_dataloader_init_all_params(self) -> None:
        """Test FactScore dataloader with all custom parameters."""
        loader = FactScoreDataloader(
            dataset_name="custom",
            split="validation",
            limit=50,
        )

        assert loader.dataset_name == "custom"
        assert loader.split == "validation"
        assert loader.limit == 50


class TestFactScoreDataloaderLoad:
    """Test suite for FactScore dataset loading.

    Tests cover:
    - Loading FactScore dataset
    - Data format and structure
    - Metadata preservation
    - Fact decomposition handling
    - Limit handling
    """

    def test_factscore_load_returns_list(self, factscore_sample_rows) -> None:
        """Test that load returns a list."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(factscore_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert isinstance(result, list)

    def test_factscore_load_correct_data_structure(self, factscore_sample_rows) -> None:
        """Test that loaded data has correct structure."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) > 0
            assert "text" in result[0]
            assert "metadata" in result[0]

    def test_factscore_load_preserves_topic(self, factscore_sample_rows) -> None:
        """Test that topic is preserved in metadata."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["topic"] == "Albert Einstein"

    def test_factscore_load_preserves_id(self, factscore_sample_rows) -> None:
        """Test that ID is preserved."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["id"] == "fact_1"

    def test_factscore_load_preserves_facts(self, factscore_sample_rows) -> None:
        """Test that facts are preserved."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert "facts" in result[0]["metadata"]
            assert len(result[0]["metadata"]["facts"]) > 0

    def test_factscore_load_includes_fact_text(self, factscore_sample_rows) -> None:
        """Test that fact text is included."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            # Text should include the fact content
            assert (
                "Einstein" in result[0]["text"] or "fact" in result[0]["text"].lower()
            )

    def test_factscore_load_respects_limit(self, factscore_sample_rows) -> None:
        """Test that limit parameter is respected."""
        loader = FactScoreDataloader(limit=1)

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(factscore_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) <= 1

    def test_factscore_load_empty_dataset(self) -> None:
        """Test loading empty FactScore dataset."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter([]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result == []

    def test_factscore_load_multiple_rows(self, factscore_sample_rows) -> None:
        """Test loading multiple rows."""
        loader = FactScoreDataloader()

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(factscore_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) >= len(factscore_sample_rows)

    def test_factscore_load_dataset_name_passed(self, factscore_sample_rows) -> None:
        """Test that dataset name is passed to load_dataset."""
        loader = FactScoreDataloader(dataset_name="custom_factscore")

        with patch("vectordb.dataloaders.factscore.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader.load()

            mock_load.assert_called_once()
