"""Tests for Haystack ARC dataset loader.

This module tests the Haystack-specific ARCDataloader implementation
which loads the ARC dataset with Haystack components.
"""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.dataloaders.haystack.arc import ARCDataloader


class TestHaystackARCDataloaderInitialization:
    """Test suite for Haystack ARCDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom dataset name
    - Custom split
    - Custom text splitter
    - Parameter handling
    """

    def test_arc_dataloader_init_defaults(self) -> None:
        """Test ARC dataloader with default parameters."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader()

            assert loader.dataset is not None
            assert loader.data is None
            assert loader.corpus == []
            assert loader.text_splitter is not None  # Default RecursiveDocumentSplitter

    def test_arc_dataloader_init_custom_split(self) -> None:
        """Test ARC dataloader with custom split."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(split="validation")

            assert loader.dataset is not None

    def test_arc_dataloader_init_custom_dataset_name(self) -> None:
        """Test ARC dataloader with custom dataset name."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(dataset_name="custom_arc")

            assert loader.dataset is not None

    def test_arc_dataloader_init_custom_text_splitter(
        self, mock_recursive_document_splitter
    ) -> None:
        """Test ARC dataloader with custom text splitter."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)

            assert loader.text_splitter == mock_recursive_document_splitter

    def test_arc_dataloader_has_text_splitter_default(self) -> None:
        """Test that text splitter defaults to RecursiveDocumentSplitter."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader()

            assert loader.text_splitter is not None


class TestHaystackARCDataloaderLoadData:
    """Test suite for Haystack ARC dataset loading.

    Tests cover:
    - Loading dataset
    - Data format and structure
    - Metadata preservation
    - Question and answer handling
    - Text splitting integration
    """

    def test_arc_load_data_returns_list(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that load_data returns a list."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result = loader.load_data()

            assert isinstance(result, list)

    def test_arc_load_data_correct_structure(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that loaded data has correct structure."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "text" in item
            assert "metadata" in item

    def test_arc_load_data_preserves_question(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that question is preserved in metadata."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "question" in metadata
            assert "What is the capital of France?" in metadata["question"]

    def test_arc_load_data_preserves_choices_and_answer(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that choices and answer are preserved in metadata."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "choices" in metadata
            assert "answer" in metadata
            assert "answerKey" in metadata

    def test_arc_load_data_preserves_context_metadata(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that context metadata is preserved."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "id" in metadata
            assert "title" in metadata

    def test_arc_load_data_caches_data(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that load_data caches the result."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result1 = loader.load_data()
            result2 = loader.load_data()

            assert result1 == result2
            # Dataset should only be iterated once
            assert mock_dataset.__iter__.call_count == 1

    def test_arc_load_data_handles_multiple_rows(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result = loader.load_data()

            assert len(result) >= len(haystack_arc_sample_rows)


class TestHaystackARCDataloaderGetDocuments:
    """Test suite for Haystack Document conversion.

    Tests cover:
    - Converting corpus to Haystack Documents
    - Document structure
    - Metadata preservation
    - Error handling
    """

    def test_arc_get_documents_returns_haystack_docs(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that get_documents returns Haystack Documents."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)

    def test_arc_get_documents_preserves_content(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that get_documents preserves content."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.content for doc in documents)

    def test_arc_get_documents_preserves_metadata(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that get_documents preserves metadata."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.meta for doc in documents)

    def test_arc_get_documents_raises_when_corpus_empty(self) -> None:
        """Test that get_documents raises error when corpus is empty."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader()

            with pytest.raises(ValueError, match="Corpus empty"):
                loader.get_documents()


class TestHaystackARCDataloaderIntegration:
    """Test suite for full ARC loader workflow.

    Tests cover:
    - Complete load and conversion pipeline
    - Data consistency
    - Multiple items handling
    """

    def test_arc_full_workflow(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test complete load_data -> get_documents workflow."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            corpus = loader.load_data()
            documents = loader.get_documents()

            assert len(corpus) > 0
            assert len(documents) > 0
            assert len(documents) == len(corpus)

    def test_arc_handles_multiple_rows(
        self, haystack_arc_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch("vectordb.dataloaders.haystack.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_recursive_document_splitter)
            result = loader.load_data()

            assert len(result) >= len(haystack_arc_sample_rows)
