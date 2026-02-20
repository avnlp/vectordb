"""Tests for namespace embedding utilities.

This module tests the embedding utilities for namespace pipelines,
including document/text embedder creation and embedding truncation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from haystack import Document

from vectordb.haystack.namespaces.utils.embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    get_document_embedder,
    get_text_embedder,
    truncate_embeddings,
)


class TestGetDocumentEmbedder:
    """Tests for get_document_embedder function."""

    def test_default_embedder_creation(self) -> None:
        """Test creating document embedder with default configuration."""
        config: dict = {}

        with patch(
            "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_document_embedder(config)

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=True,
            )
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance

    def test_custom_model_config(self) -> None:
        """Test creating document embedder with custom model."""
        config = {"embedding": {"model": "custom/model-name"}}

        with patch(
            "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_document_embedder(config)

            mock_class.assert_called_once_with(
                model="custom/model-name",
                trust_remote_code=True,
            )
            assert result == mock_instance

    def test_embedder_warm_up_called(self) -> None:
        """Test that warm_up is called on the embedder instance."""
        config = {"embedding": {"model": "test-model"}}

        with patch(
            "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            get_document_embedder(config)

            mock_instance.warm_up.assert_called_once()

    def test_default_model_constant(self) -> None:
        """Test that default model constant is correctly set."""
        assert DEFAULT_EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"

    def test_default_dimension_constant(self) -> None:
        """Test that default dimension constant is correctly set."""
        assert DEFAULT_EMBEDDING_DIMENSION == 1024


class TestGetTextEmbedder:
    """Tests for get_text_embedder function."""

    def test_default_text_embedder_creation(self) -> None:
        """Test creating text embedder with default configuration."""
        config: dict = {}

        with patch(
            "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_text_embedder(config)

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=True,
            )
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance

    def test_custom_model_text_embedder(self) -> None:
        """Test creating text embedder with custom model."""
        config = {"embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}

        with patch(
            "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_text_embedder(config)

            mock_class.assert_called_once_with(
                model="sentence-transformers/all-MiniLM-L6-v2",
                trust_remote_code=True,
            )
            assert result == mock_instance

    def test_text_embedder_warm_up_called(self) -> None:
        """Test that warm_up is called on the text embedder instance."""
        config = {"embedding": {"model": "test-model"}}

        with patch(
            "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            get_text_embedder(config)

            mock_instance.warm_up.assert_called_once()

    def test_text_embedder_with_empty_config(self) -> None:
        """Test creating text embedder with empty config."""
        config: dict = {}

        with patch(
            "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_text_embedder(config)

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=True,
            )
            assert result == mock_instance


class TestTruncateEmbeddings:
    """Tests for truncate_embeddings function."""

    def test_no_truncation_when_dimension_is_none(self) -> None:
        """Test that no truncation occurs when output_dimension is None."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
            Document(content="Doc 2", id="2", embedding=[5.0, 6.0, 7.0, 8.0]),
        ]

        result = truncate_embeddings(docs, None)

        assert result == docs
        assert result[0].embedding == [1.0, 2.0, 3.0, 4.0]
        assert result[1].embedding == [5.0, 6.0, 7.0, 8.0]

    def test_truncate_to_smaller_dimension(self) -> None:
        """Test truncation to a smaller dimension."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
            Document(content="Doc 2", id="2", embedding=[5.0, 6.0, 7.0, 8.0]),
        ]

        result = truncate_embeddings(docs, 2)

        assert result[0].embedding == [1.0, 2.0]
        assert result[1].embedding == [5.0, 6.0]

    def test_truncate_to_larger_dimension(self) -> None:
        """Test truncation to larger dimension (should keep original)."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
        ]

        result = truncate_embeddings(docs, 8)

        assert result[0].embedding == [1.0, 2.0, 3.0, 4.0]

    def test_truncate_with_none_embeddings(self) -> None:
        """Test truncation handles documents with None embeddings gracefully."""
        docs = [
            Document(content="Doc 1", id="1", embedding=None),
            Document(content="Doc 2", id="2", embedding=[1.0, 2.0, 3.0, 4.0]),
        ]

        result = truncate_embeddings(docs, 2)

        assert result[0].embedding is None
        assert result[1].embedding == [1.0, 2.0]

    def test_truncate_all_none_embeddings(self) -> None:
        """Test truncation when all documents have None embeddings."""
        docs = [
            Document(content="Doc 1", id="1", embedding=None),
            Document(content="Doc 2", id="2", embedding=None),
        ]

        result = truncate_embeddings(docs, 2)

        assert result[0].embedding is None
        assert result[1].embedding is None

    def test_truncate_empty_document_list(self) -> None:
        """Test truncation with empty document list."""
        result = truncate_embeddings([], 2)

        assert result == []

    def test_truncate_to_zero_dimension(self) -> None:
        """Test truncation to zero dimension."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
        ]

        result = truncate_embeddings(docs, 0)

        assert result[0].embedding == []

    def test_truncate_to_exact_dimension(self) -> None:
        """Test truncation when output dimension equals embedding dimension."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
        ]

        result = truncate_embeddings(docs, 4)

        assert result[0].embedding == [1.0, 2.0, 3.0, 4.0]

    def test_truncate_mixed_embeddings(self) -> None:
        """Test truncation with mixed None and valid embeddings."""
        docs = [
            Document(content="Doc 1", id="1", embedding=None),
            Document(content="Doc 2", id="2", embedding=[1.0, 2.0, 3.0]),
            Document(content="Doc 3", id="3", embedding=None),
            Document(content="Doc 4", id="4", embedding=[4.0, 5.0, 6.0, 7.0, 8.0]),
        ]

        result = truncate_embeddings(docs, 2)

        assert result[0].embedding is None
        assert result[1].embedding == [1.0, 2.0]
        assert result[2].embedding is None
        assert result[3].embedding == [4.0, 5.0]

    def test_truncate_returns_same_list(self) -> None:
        """Test truncate_embeddings returns same list object (modified in-place)."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0]),
        ]

        result = truncate_embeddings(docs, 2)

        assert result is docs
