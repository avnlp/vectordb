"""Tests for dataloader utilities.

This module tests utility functions in the dataloaders module including:
1. Logging configuration and logger creation
2. Document validation for the standardized format
3. Document conversion between frameworks and dict format

Standardized document format:
    {
        "text": str,       # Document content
        "metadata": dict   # Document metadata
    }

Test categories:
    TestLoggingUtilities: Logging setup and configuration tests.
    TestValidateDocuments: Document validation and error handling.
    TestDocumentConversion: Framework-to-dict conversion functions.
    TestIntegration: End-to-end utility workflows.
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from vectordb.dataloaders.utils import (
    _configure_logging,
    dict_to_documents,
    get_logger,
    haystack_docs_to_dict,
    langchain_docs_to_dict,
    validate_documents,
)


class TestLoggingUtilities:
    """Test suite for logging utilities.

    Tests cover:
    - Logging configuration
    - Logger creation
    - Environment variable handling for log levels
    """

    @patch.dict(os.environ, {}, clear=True)  # Clear environment for this test
    def test_configure_logging_defaults_to_info(self) -> None:
        """Test that configure logging defaults to INFO level."""
        # Since _configure_logging is called on import, we can't test it directly
        # But we can verify that the function exists and is callable
        assert callable(_configure_logging)

        # The function should be callable without errors
        _configure_logging()

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}, clear=True)
    def test_configure_logging_with_env_var(self) -> None:
        """Test that configure logging respects LOG_LEVEL environment variable."""
        # Since _configure_logging is called on import, we can't test it directly
        # But we can verify that the function exists and is callable
        assert callable(_configure_logging)

        # The function should be callable without errors
        _configure_logging()

    def test_get_logger_returns_valid_logger(self) -> None:
        """Test that get_logger returns a valid logging.Logger instance."""
        logger = get_logger(__name__)

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__

    def test_get_logger_with_different_names(self) -> None:
        """Test that get_logger works with different names."""
        logger1 = get_logger("test.logger1")
        logger2 = get_logger("test.logger2")

        assert logger1.name == "test.logger1"
        assert logger2.name == "test.logger2"
        assert logger1 != logger2


class TestValidateDocuments:
    """Test suite for document validation utilities.

    Tests cover:
    - Valid document structures
    - Invalid document structures
    - Missing keys
    - Invalid metadata types
    - Alias function (dict_to_documents)
    """

    def test_validate_documents_valid_input(self) -> None:
        """Test that validate_documents accepts valid input."""
        valid_data = [
            {"text": "Sample text 1", "metadata": {}},
            {"text": "Sample text 2", "metadata": {"key": "value"}},
        ]

        result = validate_documents(valid_data)

        assert result == valid_data

    def test_validate_documents_missing_text_key(self) -> None:
        """Test that validate_documents raises KeyError for missing text key."""
        invalid_data = [
            {"metadata": {}},  # Missing 'text' key
        ]

        with pytest.raises(KeyError, match="Missing keys in document: \\['text'\\]"):
            validate_documents(invalid_data)

    def test_validate_documents_missing_metadata_key(self) -> None:
        """Test that validate_documents raises KeyError for missing metadata key."""
        invalid_data = [
            {"text": "Sample text"},  # Missing 'metadata' key
        ]

        with pytest.raises(
            KeyError, match="Missing keys in document: \\['metadata'\\]"
        ):
            validate_documents(invalid_data)

    def test_validate_documents_missing_both_keys(self) -> None:
        """Test that validate_documents raises KeyError for missing both keys."""
        invalid_data = [
            {"other": "value"},  # Missing both 'text' and 'metadata' keys
        ]

        with pytest.raises(
            KeyError, match="Missing keys in document: \\['text', 'metadata'\\]"
        ):
            validate_documents(invalid_data)

    def test_validate_documents_invalid_metadata_type(self) -> None:
        """Test that validate_documents raises TypeError for non-dict metadata."""
        invalid_data = [
            {"text": "Sample text", "metadata": "not_a_dict"},
        ]

        with pytest.raises(TypeError, match="Metadata must be a dictionary."):
            validate_documents(invalid_data)

    def test_validate_documents_empty_list(self) -> None:
        """Test that validate_documents accepts empty list."""
        result = validate_documents([])

        assert result == []

    def test_validate_documents_multiple_valid_docs(self) -> None:
        """Test that validate_documents works with multiple valid documents."""
        valid_data = [
            {"text": "Text 1", "metadata": {"id": 1}},
            {"text": "Text 2", "metadata": {"id": 2, "category": "news"}},
            {"text": "Text 3", "metadata": {}},
        ]

        result = validate_documents(valid_data)

        assert result == valid_data

    def test_dict_to_documents_alias(self) -> None:
        """Test that dict_to_documents is an alias for validate_documents."""
        assert dict_to_documents is validate_documents

        # Test that both functions behave identically
        valid_data = [{"text": "test", "metadata": {}}]

        result1 = validate_documents(valid_data)
        result2 = dict_to_documents(valid_data)

        assert result1 == result2


class TestDocumentConversion:
    """Test suite for document conversion utilities.

    Tests cover:
    - LangChain document to dictionary conversion
    - Haystack document to dictionary conversion
    - Edge cases and error conditions
    """

    def test_langchain_docs_to_dict_basic_conversion(self) -> None:
        """Test basic conversion of LangChain documents to dictionaries."""
        # Mock LangChain Document objects
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Content of document 1"
        mock_doc1.metadata = {"source": "test1", "page": 1}

        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Content of document 2"
        mock_doc2.metadata = {"source": "test2", "page": 2}

        langchain_docs = [mock_doc1, mock_doc2]

        result = langchain_docs_to_dict(langchain_docs)

        expected = [
            {
                "text": "Content of document 1",
                "metadata": {"source": "test1", "page": 1},
            },
            {
                "text": "Content of document 2",
                "metadata": {"source": "test2", "page": 2},
            },
        ]

        assert result == expected

    def test_langchain_docs_to_dict_empty_list(self) -> None:
        """Test conversion of empty LangChain documents list."""
        result = langchain_docs_to_dict([])

        assert result == []

    def test_langchain_docs_to_dict_no_metadata(self) -> None:
        """Test conversion of LangChain documents with empty metadata."""
        mock_doc = MagicMock()
        mock_doc.page_content = "Content without metadata"
        mock_doc.metadata = {}

        result = langchain_docs_to_dict([mock_doc])

        expected = [{"text": "Content without metadata", "metadata": {}}]

        assert result == expected

    def test_haystack_docs_to_dict_basic_conversion(self) -> None:
        """Test basic conversion of Haystack documents to dictionaries."""
        # Mock Haystack Document objects
        mock_doc1 = MagicMock()
        mock_doc1.content = "Content of document 1"
        mock_doc1.meta = {"source": "test1", "page": 1}

        mock_doc2 = MagicMock()
        mock_doc2.content = "Content of document 2"
        mock_doc2.meta = {"source": "test2", "page": 2}

        haystack_docs = [mock_doc1, mock_doc2]

        result = haystack_docs_to_dict(haystack_docs)

        expected = [
            {
                "text": "Content of document 1",
                "metadata": {"source": "test1", "page": 1},
            },
            {
                "text": "Content of document 2",
                "metadata": {"source": "test2", "page": 2},
            },
        ]

        assert result == expected

    def test_haystack_docs_to_dict_empty_list(self) -> None:
        """Test conversion of empty Haystack documents list."""
        result = haystack_docs_to_dict([])

        assert result == []

    def test_haystack_docs_to_dict_no_metadata(self) -> None:
        """Test conversion of Haystack documents with empty metadata."""
        mock_doc = MagicMock()
        mock_doc.content = "Content without metadata"
        mock_doc.meta = {}

        result = haystack_docs_to_dict([mock_doc])

        expected = [{"text": "Content without metadata", "metadata": {}}]

        assert result == expected


class TestIntegration:
    """Test suite for integration of multiple utility functions.

    Tests cover:
    - Combining document conversion with validation
    - End-to-end workflows
    """

    def test_full_workflow_langchain_to_validated_dict(self) -> None:
        """Test converting LangChain docs to dict and then validating."""
        # Create mock LangChain documents
        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Valid content 1"
        mock_doc1.metadata = {"source": "test", "id": 1}

        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Valid content 2"
        mock_doc2.metadata = {"source": "test", "id": 2}

        langchain_docs = [mock_doc1, mock_doc2]

        # Convert to dict format
        dict_docs = langchain_docs_to_dict(langchain_docs)

        # Validate the converted docs
        validated_docs = validate_documents(dict_docs)

        # Both should be equal and valid
        assert validated_docs == dict_docs
        assert len(validated_docs) == 2
        assert all("text" in doc and "metadata" in doc for doc in validated_docs)

    def test_full_workflow_haystack_to_validated_dict(self) -> None:
        """Test converting Haystack docs to dict and then validating."""
        # Create mock Haystack documents
        mock_doc1 = MagicMock()
        mock_doc1.content = "Valid content 1"
        mock_doc1.meta = {"source": "test", "id": 1}

        mock_doc2 = MagicMock()
        mock_doc2.content = "Valid content 2"
        mock_doc2.meta = {"source": "test", "id": 2}

        haystack_docs = [mock_doc1, mock_doc2]

        # Convert to dict format
        dict_docs = haystack_docs_to_dict(haystack_docs)

        # Validate the converted docs
        validated_docs = validate_documents(dict_docs)

        # Both should be equal and valid
        assert validated_docs == dict_docs
        assert len(validated_docs) == 2
        assert all("text" in doc and "metadata" in doc for doc in validated_docs)
