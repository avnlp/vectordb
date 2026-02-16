"""Tests for Haystack TriviaQA dataset loader.

This module tests the Haystack-specific TriviaQADataloader implementation
which loads the TriviaQA dataset with Haystack components.
"""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.dataloaders.haystack.triviaqa import TriviaQADataloader


class TestHaystackTriviaQADataloaderInitialization:
    """Test suite for Haystack TriviaQADataloader initialization.

    Tests cover:
    - Default configuration
    - Custom dataset name
    - Custom split
    - Custom text splitter
    - Parameter handling
    """

    def test_triviaqa_dataloader_init_defaults(
        self, mock_openai_chat_generator
    ) -> None:
        """Test TriviaQA dataloader with default parameters."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator
            )

            assert loader.dataset is not None
            assert loader.data is None
            assert loader.corpus == []
            assert loader.answer_summary_generator == mock_openai_chat_generator
            assert loader.text_splitter is not None  # Default RecursiveDocumentSplitter

    def test_triviaqa_dataloader_init_custom_split(
        self, mock_openai_chat_generator
    ) -> None:
        """Test TriviaQA dataloader with custom split."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator, split="validation"
            )

            assert loader.dataset is not None

    def test_triviaqa_dataloader_init_custom_dataset_name(
        self, mock_openai_chat_generator
    ) -> None:
        """Test TriviaQA dataloader with custom dataset name."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                dataset_name="custom_triviaqa",
            )

            assert loader.dataset is not None

    def test_triviaqa_dataloader_init_custom_text_splitter(
        self, mock_recursive_document_splitter, mock_openai_chat_generator
    ) -> None:
        """Test TriviaQA dataloader with custom text splitter."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )

            assert loader.text_splitter == mock_recursive_document_splitter

    def test_triviaqa_dataloader_has_text_splitter_default(
        self, mock_openai_chat_generator
    ) -> None:
        """Test that text splitter defaults to RecursiveDocumentSplitter."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator
            )

            assert loader.text_splitter is not None


class TestHaystackTriviaQADataloaderLoadData:
    """Test suite for Haystack TriviaQA dataset loading.

    Tests cover:
    - Loading dataset
    - Data format and structure
    - Metadata preservation
    - Question and answer handling
    - Text splitting integration
    """

    def test_triviaqa_load_data_returns_list(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that load_data returns a list."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result = loader.load_data()

            assert isinstance(result, list)

    def test_triviaqa_load_data_correct_structure(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that loaded data has correct structure."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "text" in item
            assert "metadata" in item

    def test_triviaqa_load_data_preserves_question(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that question is preserved in metadata."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "question" in metadata
            assert "What is the capital of France?" in metadata["question"]

    def test_triviaqa_load_data_preserves_answers_and_summarized_answer(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that answers and summarized answer are preserved in metadata."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "answers" in metadata
            assert "answer" in metadata  # This is the summarized answer

    def test_triviaqa_load_data_preserves_context_metadata(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that context metadata is preserved."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "id" in metadata
            assert "title" in metadata

    def test_triviaqa_load_data_caches_data(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that load_data caches the result."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result1 = loader.load_data()
            result2 = loader.load_data()

            assert result1 == result2
            # Dataset should only be iterated once
            assert mock_dataset.__iter__.call_count == 1

    def test_triviaqa_load_data_handles_multiple_rows(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result = loader.load_data()

            assert len(result) >= len(haystack_triviaqa_sample_rows)


class TestHaystackTriviaQADataloaderGetDocuments:
    """Test suite for Haystack Document conversion.

    Tests cover:
    - Converting corpus to Haystack Documents
    - Document structure
    - Metadata preservation
    - Error handling
    """

    def test_triviaqa_get_documents_returns_haystack_docs(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that get_documents returns Haystack Documents."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)

    def test_triviaqa_get_documents_preserves_content(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that get_documents preserves content."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.content for doc in documents)

    def test_triviaqa_get_documents_preserves_metadata(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test that get_documents preserves metadata."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.meta for doc in documents)

    def test_triviaqa_get_documents_raises_when_corpus_empty(
        self, mock_openai_chat_generator
    ) -> None:
        """Test that get_documents raises error when corpus is empty."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator
            )

            with pytest.raises(ValueError, match="Corpus empty"):
                loader.get_documents()


class TestHaystackTriviaQADataloaderIntegration:
    """Test suite for full TriviaQA loader workflow.

    Tests cover:
    - Complete load and conversion pipeline
    - Data consistency
    - Multiple items handling
    """

    def test_triviaqa_full_workflow(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test complete load_data -> get_documents workflow."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            corpus = loader.load_data()
            documents = loader.get_documents()

            assert len(corpus) > 0
            assert len(documents) > 0
            assert len(documents) == len(corpus)

    def test_triviaqa_handles_multiple_rows(
        self,
        haystack_triviaqa_sample_rows,
        mock_recursive_document_splitter,
        mock_openai_chat_generator,
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch("vectordb.dataloaders.haystack.triviaqa.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_triviaqa_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = TriviaQADataloader(
                answer_summary_generator=mock_openai_chat_generator,
                text_splitter=mock_recursive_document_splitter,
            )
            result = loader.load_data()

            assert len(result) >= len(haystack_triviaqa_sample_rows)
