"""Tests for Earnings Calls dataset loader.

This module tests the EarningsCallDataloader class which loads
EDGAR/Earnings Call financial documents and QA pairs.
"""

from unittest.mock import MagicMock, patch

from vectordb.dataloaders.earnings_calls import EarningsCallDataloader


class TestEarningsCallDataloaderInitialization:
    """Test suite for EarningsCallDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom configuration
    - Parameter handling
    """

    def test_earnings_call_dataloader_init_defaults(self) -> None:
        """Test EarningsCall dataloader with default parameters."""
        loader = EarningsCallDataloader()

        assert loader.dataset_name == "lamini/earnings-calls-qa"
        assert loader.split == "train"

    def test_earnings_call_dataloader_init_custom_split(self) -> None:
        """Test EarningsCall dataloader with custom split."""
        loader = EarningsCallDataloader(split="validation")

        assert loader.split == "validation"

    def test_earnings_call_dataloader_init_custom_dataset_name(self) -> None:
        """Test EarningsCall dataloader with custom dataset name."""
        loader = EarningsCallDataloader(dataset_name="custom_earnings")

        assert loader.dataset_name == "custom_earnings"

    def test_earnings_call_dataloader_init_with_limit(self) -> None:
        """Test EarningsCall dataloader with limit."""
        loader = EarningsCallDataloader(limit=50)

        assert loader.limit == 50

    def test_earnings_call_dataloader_init_all_params(self) -> None:
        """Test EarningsCall dataloader with all custom parameters."""
        loader = EarningsCallDataloader(
            dataset_name="custom",
            split="validation",
            limit=50,
        )

        assert loader.dataset_name == "custom"
        assert loader.split == "validation"
        assert loader.limit == 50


class TestEarningsCallDataloaderLoad:
    """Test suite for Earnings Call dataset loading.

    Tests cover:
    - Loading earnings call dataset
    - Data format and structure
    - Metadata preservation
    - Financial document handling
    - Limit handling
    """

    def test_earnings_call_load_returns_list(self, earnings_calls_sample_rows) -> None:
        """Test that load returns a list."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert isinstance(result, list)

    def test_earnings_call_load_correct_data_structure(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that loaded data has correct structure."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) > 0
            assert "text" in result[0]
            assert "metadata" in result[0]

    def test_earnings_call_load_preserves_company(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that company name is preserved in metadata."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["company"] == "Acme Corp"

    def test_earnings_call_load_preserves_date(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that date is preserved in metadata."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["date"] == "2024-01-15"

    def test_earnings_call_load_preserves_quarter(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that quarter is preserved in metadata."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["quarter"] == "Q4"

    def test_earnings_call_load_preserves_year(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that year is preserved in metadata."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result[0]["metadata"]["year"] == 2023

    def test_earnings_call_load_includes_content(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that content is included in text field."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert "earnings call" in result[0]["text"].lower()

    def test_earnings_call_load_respects_limit(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that limit parameter is respected."""
        loader = EarningsCallDataloader(limit=1)

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) == 1

    def test_earnings_call_load_empty_dataset(self) -> None:
        """Test loading empty earnings call dataset."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter([]))
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert result == []

    def test_earnings_call_load_multiple_rows(self, earnings_calls_sample_rows) -> None:
        """Test loading multiple rows."""
        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            result = loader.load()

            assert len(result) == len(earnings_calls_sample_rows)

    def test_earnings_call_load_dataset_name_passed(
        self, earnings_calls_sample_rows
    ) -> None:
        """Test that dataset name is passed to load_dataset."""
        loader = EarningsCallDataloader(dataset_name="custom_earnings")

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader.load()

            mock_load.assert_called_once()

    def test_earnings_call_load_qa_format(self) -> None:
        """Test loading earnings calls in QA format."""
        qa_rows = [
            {
                "question": "What was revenue?",
                "answer": "$100M",
                "transcript": "Revenue was $100M",
                "date": "2024-01-15",
                "q": "2024-Q1",
                "ticker": "ACME",
            },
            {
                "question": "What was EPS?",
                "answer": "$2.50",
                "transcript": "EPS was $2.50",
                "date": "2024-01-15",
                "q": "2024-Q1",
                "ticker": "ACME",
            },
        ]

        loader = EarningsCallDataloader()

        with patch("vectordb.dataloaders.earnings_calls.hf_load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(qa_rows))
            mock_load.return_value = mock_dataset

            result = loader.load()

            # Should handle QA format
            assert len(result) > 0
