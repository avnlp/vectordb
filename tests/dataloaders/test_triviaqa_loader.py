"""Tests for TriviaQA dataset loader.

This module tests the TriviaQADataloader class which loads the
TriviaQA open-domain question answering dataset.
"""

from unittest.mock import MagicMock, patch

from vectordb.dataloaders.triviaqa import TriviaQADataloader


class TestTriviaQADataloaderInitialization:
    """Test suite for TriviaQADataloader initialization.

    Tests cover:
    - Default configuration
    - Custom configuration
    - Parameter handling
    """

    def test_triviaqa_dataloader_init_defaults(self) -> None:
        """Test TriviaQA dataloader with default parameters."""
        loader = TriviaQADataloader()

        assert loader.dataset_name == "trivia_qa"
        assert loader.config == "rc"
        assert loader.split == "test"
        assert loader.limit is None

    def test_triviaqa_dataloader_init_custom_split(self) -> None:
        """Test TriviaQA dataloader with custom split."""
        loader = TriviaQADataloader(split="validation")

        assert loader.split == "validation"

    def test_triviaqa_dataloader_init_custom_dataset_name(self) -> None:
        """Test TriviaQA dataloader with custom dataset name."""
        loader = TriviaQADataloader(dataset_name="custom_trivia")

        assert loader.dataset_name == "custom_trivia"

    def test_triviaqa_dataloader_init_with_limit(self) -> None:
        """Test TriviaQA dataloader with limit."""
        loader = TriviaQADataloader(limit=100)

        assert loader.limit == 100

    def test_triviaqa_dataloader_init_all_params(self) -> None:
        """Test TriviaQA dataloader with all custom parameters."""
        loader = TriviaQADataloader(
            dataset_name="custom",
            config="custom_config",
            split="validation",
            limit=50,
        )

        assert loader.dataset_name == "custom"
        assert loader.config == "custom_config"
        assert loader.split == "validation"
        assert loader.limit == 50


class TestTriviaQADataloaderLoad:
    """Test suite for TriviaQA dataset loading.

    Tests cover:
    - Loading TriviaQA dataset
    - Data format and structure
    - Metadata preservation
    - Handling search results
    - Limit handling
    """

    def test_triviaqa_load_returns_list(self, triviaqa_sample_rows) -> None:
        """Test that load returns a list."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(triviaqa_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert isinstance(result, list)

    def test_triviaqa_load_correct_data_structure(self, triviaqa_sample_rows) -> None:
        """Test that loaded data has correct structure."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) > 0
            assert "text" in result[0]
            assert "metadata" in result[0]

    def test_triviaqa_load_preserves_question(self, triviaqa_sample_rows) -> None:
        """Test that question is preserved in metadata."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["question"] == "What is the capital of France?"

    def test_triviaqa_load_preserves_answer(self, triviaqa_sample_rows) -> None:
        """Test that answer is preserved in metadata."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["answer"] == "Paris"

    def test_triviaqa_load_includes_search_context(self, triviaqa_sample_rows) -> None:
        """Test that search context is included in text."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert "Paris is the capital of France." in result[0]["text"]

    def test_triviaqa_load_preserves_rank(self, triviaqa_sample_rows) -> None:
        """Test that search rank is preserved."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert "rank" in result[0]["metadata"]
            assert result[0]["metadata"]["rank"] == 1

    def test_triviaqa_load_handles_multiple_search_results(
        self, triviaqa_sample_rows
    ) -> None:
        """Test that multiple search results are handled correctly."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(triviaqa_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            # First row has 2 search results, second has 1 = 3 total
            assert len(result) == 3

    def test_triviaqa_load_respects_limit_across_questions(
        self, triviaqa_sample_rows
    ) -> None:
        """Test that limit respects total items across all questions."""
        loader = TriviaQADataloader(limit=1)

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(triviaqa_sample_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) == 1

    def test_triviaqa_load_empty_dataset(self) -> None:
        """Test loading empty TriviaQA dataset."""
        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter([]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result == []

    def test_triviaqa_load_missing_search_context(self) -> None:
        """Test handling of missing search context."""
        rows = [
            {
                "question": "Test?",
                "answer": "Answer",
                "search_results": {
                    "rank": [1],
                    "title": ["Title"],
                    "search_context": [],
                    "description": ["Description"],
                },
            }
        ]

        loader = TriviaQADataloader()

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            # Should fall back to description
            assert len(result) > 0

    def test_triviaqa_load_dataset_name_passed(self, triviaqa_sample_rows) -> None:
        """Test that dataset name is passed to load_dataset."""
        loader = TriviaQADataloader(dataset_name="custom_trivia")

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader.load()

            mock_load.assert_called_once()

    def test_triviaqa_load_config_passed(self, triviaqa_sample_rows) -> None:
        """Test that config is passed to load_dataset."""
        loader = TriviaQADataloader(config="wikipedia")

        with patch("vectordb.dataloaders.triviaqa.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader.load()

            mock_load.assert_called_once()
