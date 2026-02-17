"""Unit tests for TriviaQA loader."""

from unittest.mock import patch

from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.datasets.triviaqa import TriviaQALoader


class TestTriviaQALoaderInitialization:
    """Tests for TriviaQA loader initialization."""

    def test_defaults(self) -> None:
        """Test that loader uses correct default values for all parameters."""
        loader = TriviaQALoader()

        assert loader.dataset_name == "trivia_qa"
        assert loader.config == "rc"
        assert loader.split == "test"
        assert loader.limit is None
        assert loader.streaming is True

    def test_custom_values(self) -> None:
        """Test that loader correctly applies custom parameter values."""
        loader = TriviaQALoader(
            dataset_name="custom",
            config="unfiltered",
            split="train",
            limit=5,
            streaming=False,
        )

        assert loader.dataset_name == "custom"
        assert loader.config == "unfiltered"
        assert loader.split == "train"
        assert loader.limit == 5
        assert loader.streaming is False


class TestTriviaQALoaderLoad:
    """Tests for TriviaQA loader load behavior."""

    def test_load_returns_loaded_dataset(
        self, triviaqa_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that load() returns a LoadedDataset instance."""
        loader = TriviaQALoader()

        with patch(
            "vectordb.dataloaders.datasets.triviaqa.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(triviaqa_sample_rows)
            dataset = loader.load()

        assert isinstance(dataset, LoadedDataset)

    def test_row_expansion(self, triviaqa_sample_rows, make_streaming_dataset) -> None:
        """Test that each TriviaQA row expands to one record per answer source."""
        loader = TriviaQALoader()

        with patch(
            "vectordb.dataloaders.datasets.triviaqa.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(triviaqa_sample_rows[:1])
            dataset = loader.load()

        assert len(dataset.records()) == 2

    def test_context_precedence(
        self, triviaqa_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that context field takes precedence over description when present."""
        loader = TriviaQALoader()

        with patch(
            "vectordb.dataloaders.datasets.triviaqa.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(triviaqa_sample_rows[:1])
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.text == "Paris is the capital of France."

    def test_description_fallback(
        self, triviaqa_edge_missing_context, make_streaming_dataset
    ) -> None:
        """Test that description is used as fallback when context is missing."""
        loader = TriviaQALoader()

        with patch(
            "vectordb.dataloaders.datasets.triviaqa.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(
                triviaqa_edge_missing_context
            )
            dataset = loader.load()

        record = dataset.records()[0]
        assert record.text == "Fallback description"

    def test_rank_and_title_bounds(
        self, triviaqa_sample_rows, make_streaming_dataset
    ) -> None:
        """Test that rank and title are correctly extracted from answer sources."""
        loader = TriviaQALoader()

        with patch(
            "vectordb.dataloaders.datasets.triviaqa.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(triviaqa_sample_rows[:1])
            dataset = loader.load()

        record = dataset.records()[1]
        assert record.metadata["rank"] == 2
        assert record.metadata["title"] == "Paris (Texas)"

    def test_record_limit(self, triviaqa_sample_rows, make_streaming_dataset) -> None:
        """Test that the limit parameter restricts the number of records returned."""
        loader = TriviaQALoader(limit=1)

        with patch(
            "vectordb.dataloaders.datasets.triviaqa.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset(triviaqa_sample_rows[:1])
            dataset = loader.load()

        assert len(dataset.records()) == 1


class TestTriviaQALoaderEdgeCases:
    """Tests for TriviaQA loader edge cases."""

    def test_empty_dataset(self, make_streaming_dataset) -> None:
        """Test that loader handles empty datasets gracefully."""
        loader = TriviaQALoader()

        with patch(
            "vectordb.dataloaders.datasets.triviaqa.hf_load_dataset"
        ) as mock_load:
            mock_load.return_value = make_streaming_dataset([])
            dataset = loader.load()

        assert dataset.records() == []
