"""Tests for LangChain FactScore dataset loader.

This module tests the LangChain-specific FactScoreDataloader implementation
which loads the FactScore dataset with LangChain components.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from vectordb.dataloaders.langchain.factscore import FactScoreDataloader


class TestLangChainFactScoreDataloaderInitialization:
    """Test suite for LangChain FactScoreDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom dataset name
    - Custom split
    - Custom text splitter
    - Parameter handling
    """

    def test_factscore_dataloader_init_defaults(self, mock_groq_generator) -> None:
        """Test FactScore dataloader with default parameters."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(answer_summary_generator=mock_groq_generator)

            assert loader.dataset is not None
            assert loader.data is None
            assert loader.corpus == []

    def test_factscore_dataloader_init_custom_split(self, mock_groq_generator) -> None:
        """Test FactScore dataloader with custom split."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator, split="validation"
            )

            assert loader.dataset is not None

    def test_factscore_dataloader_init_custom_text_splitter(
        self, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test FactScore dataloader with custom text splitter."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )

            assert loader.text_splitter == mock_text_splitter

    def test_factscore_dataloader_has_text_splitter_default(
        self, mock_groq_generator
    ) -> None:
        """Test that text splitter defaults to RecursiveCharacterTextSplitter."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(answer_summary_generator=mock_groq_generator)

            assert loader.text_splitter is not None


class TestLangChainFactScoreDataloaderLoadData:
    """Test suite for LangChain FactScore dataset loading.

    Tests cover:
    - Loading dataset
    - Data format and structure
    - Metadata preservation
    - Topic and fact handling
    - Fact decomposition
    - Text splitting integration
    """

    def test_factscore_load_data_returns_list(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that load_data returns a list."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert isinstance(result, list)

    def test_factscore_load_data_correct_structure(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that loaded data has correct structure."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "text" in item
            assert "metadata" in item

    def test_factscore_load_data_preserves_topic(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that topic is preserved in metadata."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "topic" in metadata

    def test_factscore_load_data_preserves_id(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that ID is preserved in metadata."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "id" in metadata

    def test_factscore_load_data_preserves_facts(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that facts list is preserved in metadata."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) > 0
            metadata = result[0]["metadata"]
            assert "facts" in metadata
            assert isinstance(metadata["facts"], list)

    def test_factscore_load_data_handles_decomposed_facts(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that decomposed facts are properly handled."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            # Should have expanded items for decomposed facts
            assert len(result) >= 1

    def test_factscore_load_data_caches_data(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that load_data caches the result."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result1 = loader.load_data()
            result2 = loader.load_data()

            assert result1 == result2
            # Dataset should only be iterated once
            assert mock_dataset.__iter__.call_count == 1


class TestLangChainFactScoreDataloaderGetDocuments:
    """Test suite for LangChain Document conversion.

    Tests cover:
    - Converting corpus to LangChain Documents
    - Document structure
    - Metadata preservation
    - Error handling
    """

    def test_factscore_get_documents_returns_langchain_docs(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that get_documents returns LangChain Documents."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)

    def test_factscore_get_documents_preserves_content(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that get_documents preserves page content."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.page_content for doc in documents)

    def test_factscore_get_documents_preserves_metadata(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test that get_documents preserves metadata."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows[:1])
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            loader.load_data()
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.metadata for doc in documents)

    def test_factscore_get_documents_raises_when_corpus_empty(
        self, mock_groq_generator
    ) -> None:
        """Test that get_documents raises error when corpus is empty."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(answer_summary_generator=mock_groq_generator)

            with pytest.raises(ValueError, match="Corpus empty"):
                loader.get_documents()


class TestLangChainFactScoreDataloaderIntegration:
    """Test suite for full FactScore loader workflow.

    Tests cover:
    - Complete load and conversion pipeline
    - Data consistency
    - Multiple items handling
    """

    def test_factscore_full_workflow(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test complete load_data -> get_documents workflow."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            corpus = loader.load_data()
            documents = loader.get_documents()

            assert len(corpus) > 0
            assert len(documents) > 0
            assert len(documents) == len(corpus)

    def test_factscore_handles_multiple_rows(
        self, langchain_factscore_sample_rows, mock_text_splitter, mock_groq_generator
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch(
            "vectordb.dataloaders.langchain.factscore.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(langchain_factscore_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = FactScoreDataloader(
                answer_summary_generator=mock_groq_generator,
                text_splitter=mock_text_splitter,
            )
            result = loader.load_data()

            assert len(result) >= len(langchain_factscore_sample_rows)
