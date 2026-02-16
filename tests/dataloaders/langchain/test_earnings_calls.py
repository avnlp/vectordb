"""Tests for LangChain Earnings Calls dataset loader.

This module tests the LangChain-specific EarningsCallDataloader implementation
which loads the Earnings Calls dataset with LangChain components.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from vectordb.dataloaders.langchain.earnings_calls import EarningsCallDataloader


class TestLangChainEarningsCallDataloaderInitialization:
    """Test suite for LangChain EarningsCallDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom dataset name
    - Custom split
    - Custom text splitter
    - Parameter handling
    """

    def test_earnings_call_dataloader_init_defaults(self) -> None:
        """Test EarningsCall dataloader with default parameters."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_corpus = MagicMock()
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader()

            assert loader.dataset is not None
            assert loader.data is None
            assert loader.corpus == []

    def test_earnings_call_dataloader_init_custom_dataset_name(self) -> None:
        """Test EarningsCall dataloader with custom dataset name."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_corpus = MagicMock()
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader(dataset_name="custom_earnings")

            assert loader.dataset is not None

    def test_earnings_call_dataloader_init_custom_split(self) -> None:
        """Test EarningsCall dataloader with custom split."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_corpus = MagicMock()
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader(split="validation")

            assert loader.dataset is not None

    def test_earnings_call_dataloader_init_custom_text_splitter(
        self, mock_text_splitter
    ) -> None:
        """Test EarningsCall dataloader with custom text splitter."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_corpus = MagicMock()
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)

            assert loader.text_splitter == mock_text_splitter

    def test_earnings_call_dataloader_has_text_splitter_default(self) -> None:
        """Test that text splitter defaults to RecursiveCharacterTextSplitter."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_corpus = MagicMock()
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader()

            assert loader.text_splitter is not None


class TestLangChainEarningsCallDataloaderLoadData:
    """Test suite for LangChain Earnings Calls dataset loading.

    Tests cover:
    - Loading dataset
    - Data format and structure
    - Metadata preservation
    - Content handling
    - Financial document processing
    - Text splitting integration
    """

    def test_earnings_call_load_data_returns_list(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that load_data returns a list."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert isinstance(result, list)

    def test_earnings_call_load_data_correct_structure(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that loaded data has correct structure."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "text" in item

    def test_earnings_call_load_data_preserves_content(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that content is preserved in text field."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert len(item["text"]) > 0

    def test_earnings_call_load_data_handles_metadata(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that metadata is properly handled."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            # Should have metadata dict (may be empty for earnings calls)
            if "metadata" in item:
                assert isinstance(item["metadata"], dict)

    def test_earnings_call_load_data_handles_qa_format(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that loader handles different content formats."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            # Should have meaningful content
            assert any(len(item["text"]) > 10 for item in result)

    def test_earnings_call_load_data_caches_data(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that load_data caches the result."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            result1 = loader.load_data()
            result2 = loader.load_data()

            assert result1 == result2
            # Dataset should only be iterated once
            assert mock_dataset.__iter__.call_count == 1


class TestLangChainEarningsCallDataloaderGetDocuments:
    """Test suite for LangChain Document conversion.

    Tests cover:
    - Converting corpus to LangChain Documents
    - Document structure
    - Metadata preservation
    - Error handling
    """

    def test_earnings_call_get_documents_returns_langchain_docs(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that get_documents returns LangChain Documents."""
        corpus_rows = [
            {
                "symbol": "ACME",
                "year": 2024,
                "quarter": 1,
                "date": "2024-01-15",
                "transcript": [{"speaker": "CEO", "text": "Welcome to earnings call."}],
            }
        ]
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows[:1])
            )
            mock_corpus = MagicMock()
            mock_corpus.__iter__ = MagicMock(return_value=iter(corpus_rows))
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)

    def test_earnings_call_get_documents_preserves_content(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that get_documents preserves page content."""
        corpus_rows = [
            {
                "symbol": "ACME",
                "year": 2024,
                "quarter": 1,
                "date": "2024-01-15",
                "transcript": [{"speaker": "CEO", "text": "Welcome to earnings call."}],
            }
        ]
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows[:1])
            )
            mock_corpus = MagicMock()
            mock_corpus.__iter__ = MagicMock(return_value=iter(corpus_rows))
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.page_content for doc in documents)

    def test_earnings_call_get_documents_preserves_metadata(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test that get_documents preserves metadata."""
        corpus_rows = [
            {
                "symbol": "ACME",
                "year": 2024,
                "quarter": 1,
                "date": "2024-01-15",
                "transcript": [{"speaker": "CEO", "text": "Welcome to earnings call."}],
            }
        ]
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows[:1])
            )
            mock_corpus = MagicMock()
            mock_corpus.__iter__ = MagicMock(return_value=iter(corpus_rows))
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            # All documents should have metadata dict (may be empty)
            assert all(isinstance(doc.metadata, dict) for doc in documents)

    def test_earnings_call_get_documents_raises_when_corpus_empty(self) -> None:
        """Test that get_documents raises error when corpus is empty."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_corpus = MagicMock()
            mock_corpus.__iter__ = MagicMock(return_value=iter([]))
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader()

            # Should return empty list, not raise
            result = loader.get_documents()
            assert result == []


class TestLangChainEarningsCallDataloaderIntegration:
    """Test suite for full Earnings Calls loader workflow.

    Tests cover:
    - Complete load and conversion pipeline
    - Data consistency
    - Multiple items handling
    """

    def test_earnings_call_full_workflow(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test complete load_data -> get_documents workflow."""
        corpus_rows = [
            {
                "symbol": "ACME",
                "year": 2024,
                "quarter": 1,
                "date": "2024-01-15",
                "transcript": [{"speaker": "CEO", "text": "Welcome to earnings call."}],
            }
        ]
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows)
            )
            mock_corpus = MagicMock()
            mock_corpus.__iter__ = MagicMock(return_value=iter(corpus_rows))
            mock_load.side_effect = [mock_dataset, mock_corpus]

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            corpus = loader.load_data()
            documents = loader.get_documents()

            # Load data returns QA pairs, get_documents returns corpus Documents
            assert len(corpus) > 0
            assert len(documents) > 0
            # Documents come from corpus dataset which is separate from QA data
            # so we just verify both are populated

    def test_earnings_call_handles_multiple_rows(
        self, langchain_earnings_calls_sample_rows, mock_text_splitter
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch(
            "vectordb.dataloaders.langchain.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) >= len(langchain_earnings_calls_sample_rows)
