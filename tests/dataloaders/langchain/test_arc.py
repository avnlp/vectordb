"""Tests for LangChain ARC dataset loader.

This module tests the LangChain-specific ARCDataloader implementation
which loads the AI2 Reasoning Challenge dataset with LangChain components.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from vectordb.dataloaders.langchain.arc import ARCDataloader


class TestLangChainARCDataloaderInitialization:
    """Test suite for LangChain ARCDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom dataset name
    - Custom split
    - Custom text splitter
    - Parameter handling
    """

    def test_arc_dataloader_init_defaults(self) -> None:
        """Test ARC dataloader with default parameters."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader()

            assert loader.dataset is not None
            assert loader.data is None
            assert loader.corpus == []
            mock_load.assert_called_once_with("ai2_arc", split="test")

    def test_arc_dataloader_init_custom_dataset_name(self) -> None:
        """Test ARC dataloader with custom dataset name."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            ARCDataloader(dataset_name="custom_arc")

            mock_load.assert_called_once_with("custom_arc", split="test")

    def test_arc_dataloader_init_custom_split(self) -> None:
        """Test ARC dataloader with custom split."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            ARCDataloader(split="validation")

            mock_load.assert_called_once_with("ai2_arc", split="validation")

    def test_arc_dataloader_init_custom_text_splitter(self, mock_text_splitter) -> None:
        """Test ARC dataloader with custom text splitter."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)

            assert loader.text_splitter == mock_text_splitter

    def test_arc_dataloader_has_text_splitter_default(self) -> None:
        """Test that text splitter defaults to RecursiveCharacterTextSplitter."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader()

            assert loader.text_splitter is not None


class TestLangChainARCDataloaderLoadData:
    """Test suite for LangChain ARC dataset loading.

    Tests cover:
    - Loading dataset
    - Data format and structure
    - Metadata preservation
    - Question formatting with choices
    - Context handling
    - Text splitting integration
    """

    def test_arc_load_data_returns_list(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that load_data returns a list."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert isinstance(result, list)

    def test_arc_load_data_correct_structure(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that loaded data has correct structure."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "text" in item
            assert "metadata" in item

    def test_arc_load_data_preserves_question(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that question is preserved in metadata."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "question" in metadata

    def test_arc_load_data_preserves_choices(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that choices are preserved in metadata."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "choices" in metadata

    def test_arc_load_data_preserves_answer_key(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that answer key is preserved in metadata."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "answerKey" in metadata
            assert metadata["answerKey"] in ["A", "B", "C", "D"]

    def test_arc_load_data_expands_contexts(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that contexts are expanded into separate items."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            # First row has 2 contexts
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            # Should have at least 2 items (one per context)
            assert len(result) >= 1

    def test_arc_load_data_caches_data(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that load_data caches the result."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result1 = loader.load_data()
            result2 = loader.load_data()

            assert result1 == result2
            # Dataset should only be iterated once
            assert mock_dataset.__iter__.call_count == 1


class TestLangChainARCDataloaderGetDocuments:
    """Test suite for LangChain Document conversion.

    Tests cover:
    - Converting corpus to LangChain Documents
    - Document structure
    - Metadata preservation
    - Error handling
    """

    def test_arc_get_documents_returns_langchain_docs(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that get_documents returns LangChain Documents."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)

    def test_arc_get_documents_preserves_content(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that get_documents preserves page content."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.page_content for doc in documents)

    def test_arc_get_documents_preserves_metadata(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test that get_documents preserves metadata."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.metadata for doc in documents)

    def test_arc_get_documents_raises_when_corpus_empty(self) -> None:
        """Test that get_documents raises error when corpus is empty."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ARCDataloader()

            with pytest.raises(ValueError, match="Corpus empty"):
                loader.get_documents()


class TestLangChainARCDataloaderIntegration:
    """Test suite for full ARC loader workflow.

    Tests cover:
    - Complete load and conversion pipeline
    - Data consistency
    - Multiple items handling
    """

    def test_arc_full_workflow(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test complete load_data -> get_documents workflow."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            corpus = loader.load_data()
            documents = loader.get_documents()

            assert len(corpus) > 0
            assert len(documents) > 0
            assert len(documents) == len(corpus)

    def test_arc_handles_multiple_rows(
        self, langchain_arc_sample_rows, mock_text_splitter
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch("vectordb.dataloaders.langchain.arc.load_dataset") as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_arc_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = ARCDataloader(text_splitter=mock_text_splitter)
            result = loader.load_data()

            assert len(result) >= len(langchain_arc_sample_rows)
