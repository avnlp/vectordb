"""Unit tests for document converters."""

import pytest
from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangChainDocument

from vectordb.dataloaders.converters import DocumentConverter


class TestDocumentConverterToHaystack:
    """Tests for Haystack conversion."""

    def test_mapping(self) -> None:
        """Test that dict documents are correctly mapped to Haystack Document objects.

        Verifies that text content and metadata are properly transferred from
        input dictionaries to the resulting Haystack Document instances.
        """
        docs = DocumentConverter.to_haystack([{"text": "Hello", "metadata": {"id": 1}}])

        assert isinstance(docs[0], HaystackDocument)
        assert docs[0].content == "Hello"
        assert docs[0].meta == {"id": 1}

    def test_empty_list(self) -> None:
        """Test that an empty list input returns an empty list."""
        assert DocumentConverter.to_haystack([]) == []


class TestDocumentConverterToLangChain:
    """Tests for LangChain conversion."""

    def test_mapping(self) -> None:
        """Test that dict documents are correctly mapped to LangChain Document objects.

        Verifies that text content is mapped to page_content and metadata is
        properly transferred from input dictionaries to the resulting
        LangChain Document instances.
        """
        docs = DocumentConverter.to_langchain(
            [{"text": "Hello", "metadata": {"id": 1}}]
        )

        assert isinstance(docs[0], LangChainDocument)
        assert docs[0].page_content == "Hello"
        assert docs[0].metadata == {"id": 1}

    def test_empty_list(self) -> None:
        """Test that an empty list input returns an empty list."""
        assert DocumentConverter.to_langchain([]) == []


class TestDocumentConverterValidation:
    """Tests for converter validation."""

    def test_missing_key_raises(self) -> None:
        """Test that missing required keys raise KeyError.

        Verifies that the converter properly raises a KeyError when the input
        dictionary is missing the required 'text' key.
        """
        with pytest.raises(KeyError):
            DocumentConverter.to_haystack([{"metadata": {}}])
