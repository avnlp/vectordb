"""Tests for LangChain TriviaQA dataset loader.

This module tests the LangChain-specific TriviaQADataloader implementation
which loads the TriviaQA dataset with LangChain components.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from vectordb.dataloaders.langchain.triviaqa import TriviaQADataloader


class TestLangChainTriviaQADataloaderInitialization:
    """Test suite for LangChain TriviaQADataloader initialization.

    Tests cover:
    - Default configuration
    - Custom dataset name
    - Custom split
    - Custom text splitter
    - Parameter handling
    """

    def test_triviaqa_dataloader_init_defaults(self, mock_groq_generator) -> None:
        """Test TriviaQA dataloader with default parameters."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(answer_summary_generator=mock_groq_generator)

            assert loader.dataset is not None
            assert loader.data is None
            assert loader.corpus == []

    def test_triviaqa_dataloader_init_custom_split(self, mock_groq_generator) -> None:
        """Test TriviaQA dataloader with custom split."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator, split="validation"
            )

            assert loader.dataset is not None

    def test_triviaqa_dataloader_init_custom_text_splitter(
        self, mock_groq_generator, mock_text_splitter
    ) -> None:
        """Test TriviaQA dataloader with custom text splitter."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )

            assert loader.text_splitter == mock_text_splitter

    def test_triviaqa_dataloader_has_text_splitter_default(
        self, mock_groq_generator
    ) -> None:
        """Test that text splitter defaults to RecursiveCharacterTextSplitter."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(answer_summary_generator=mock_groq_generator)

            assert loader.text_splitter is not None


class TestLangChainTriviaQADataloaderLoadData:
    """Test suite for LangChain TriviaQA dataset loading.

    Tests cover:
    - Loading dataset
    - Data format and structure
    - Metadata preservation
    - Question and answer formatting
    - Search result handling
    - Text splitting integration
    """

    def test_triviaqa_load_data_returns_list(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that load_data returns a list."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert isinstance(result, list)

    def test_triviaqa_load_data_correct_structure(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that loaded data has correct structure."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "text" in item
            assert "metadata" in item

    def test_triviaqa_load_data_preserves_question(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that question is preserved in metadata."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "question" in metadata

    def test_triviaqa_load_data_preserves_answer(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that answer is preserved in metadata."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "answer" in metadata

    def test_triviaqa_load_data_expands_search_results(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that search results are expanded into separate items."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            # First row has 2 search results
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            # Should have multiple items from search result expansion
            assert len(result) >= 1

    def test_triviaqa_load_data_preserves_rank(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that search rank is preserved in metadata."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            # Check that rank exists in at least one item
            assert any("rank" in item["metadata"] for item in result)

    def test_triviaqa_load_data_caches_data(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that load_data caches the result."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result1 = loader.load_data()
            result2 = loader.load_data()

            assert result1 == result2
            # Dataset should only be iterated once
            assert mock_dataset.__iter__.call_count == 1


class TestLangChainTriviaQADataloaderGetDocuments:
    """Test suite for LangChain Document conversion.

    Tests cover:
    - Converting corpus to LangChain Documents
    - Document structure
    - Metadata preservation
    - Error handling
    """

    def test_triviaqa_get_documents_returns_langchain_docs(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that get_documents returns LangChain Documents."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)

    def test_triviaqa_get_documents_preserves_content(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that get_documents preserves page content."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.page_content for doc in documents)

    def test_triviaqa_get_documents_preserves_metadata(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that get_documents preserves metadata."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.metadata for doc in documents)

    def test_triviaqa_get_documents_raises_when_corpus_empty(
        self, mock_groq_generator
    ) -> None:
        """Test that get_documents raises error when corpus is empty."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(answer_summary_generator=mock_groq_generator)

            with pytest.raises(ValueError, match="Corpus empty"):
                loader.get_documents()


class TestLangChainTriviaQADataloaderIntegration:
    """Test suite for full TriviaQA loader workflow.

    Tests cover:
    - Complete load and conversion pipeline
    - Data consistency
    - Multiple items handling
    """

    def test_triviaqa_full_workflow(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test complete load_data -> get_documents workflow."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            corpus = loader.load_data()
            documents = loader.get_documents()

            assert len(corpus) > 0
            assert len(documents) > 0
            assert len(documents) == len(corpus)

    def test_triviaqa_handles_multiple_rows(
        self, langchain_triviaqa_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch("vectordb.dataloaders.langchain.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) >= len(langchain_triviaqa_sample_rows)
