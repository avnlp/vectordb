"""Tests for multi-tenancy embedding utilities."""

from unittest.mock import MagicMock, patch

from haystack import Document

from vectordb.haystack.multi_tenancy.common.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    create_document_embedder,
    create_text_embedder,
    truncate_embeddings,
)


class TestCreateDocumentEmbedder:
    """Tests for create_document_embedder function."""

    def test_default_embedder(self):
        """Test creating embedder with default config."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = create_document_embedder()

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=True,
                batch_size=32,
            )
            assert result == mock_instance

    def test_custom_model_config(self):
        """Test creating embedder with custom model."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {"embedding": {"model": "custom/model"}}
            result = create_document_embedder(config)

            mock_class.assert_called_once_with(
                model="custom/model",
                trust_remote_code=True,
                batch_size=32,
            )
            assert result == mock_instance

    def test_custom_batch_size(self):
        """Test creating embedder with custom batch size."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {"embedding": {"batch_size": 64}}
            result = create_document_embedder(config)

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=True,
                batch_size=64,
            )
            assert result == mock_instance

    def test_custom_trust_remote_code(self):
        """Test creating embedder with custom trust_remote_code."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {"embedding": {"trust_remote_code": False}}
            result = create_document_embedder(config)

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=False,
                batch_size=32,
            )
            assert result == mock_instance


class TestCreateTextEmbedder:
    """Tests for create_text_embedder function."""

    def test_default_embedder(self):
        """Test creating text embedder with default config."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = create_text_embedder()

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=True,
                prefix="",
            )
            assert result == mock_instance

    def test_custom_model_config(self):
        """Test creating text embedder with custom model."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {"embedding": {"model": "custom/text-model"}}
            result = create_text_embedder(config)

            mock_class.assert_called_once_with(
                model="custom/text-model",
                trust_remote_code=True,
                prefix="",
            )
            assert result == mock_instance

    def test_custom_query_prefix(self):
        """Test creating text embedder with custom query prefix."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {"embedding": {"query_prefix": "query:"}}
            result = create_text_embedder(config)

            mock_class.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL,
                trust_remote_code=True,
                prefix="query:",
            )
            assert result == mock_instance


class TestTruncateEmbeddings:
    """Tests for truncate_embeddings function."""

    def test_no_truncation_when_none(self):
        """Test that no truncation occurs when output_dimension is None."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
            Document(content="Doc 2", id="2", embedding=[5.0, 6.0, 7.0, 8.0]),
        ]
        result = truncate_embeddings(docs, None)
        assert result == docs
        assert result[0].embedding == [1.0, 2.0, 3.0, 4.0]

    def test_truncate_to_smaller_dimension(self):
        """Test truncation to smaller dimension."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
            Document(content="Doc 2", id="2", embedding=[5.0, 6.0, 7.0, 8.0]),
        ]
        result = truncate_embeddings(docs, 2)
        assert result[0].embedding == [1.0, 2.0]
        assert result[1].embedding == [5.0, 6.0]

    def test_truncate_to_larger_dimension(self):
        """Test truncation to larger dimension (no effect)."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
        ]
        result = truncate_embeddings(docs, 8)
        assert result[0].embedding == [1.0, 2.0, 3.0, 4.0]

    def test_truncate_with_none_embedding(self):
        """Test truncation handles None embeddings."""
        docs = [
            Document(content="Doc 1", id="1", embedding=None),
            Document(content="Doc 2", id="2", embedding=[1.0, 2.0, 3.0, 4.0]),
        ]
        result = truncate_embeddings(docs, 2)
        assert result[0].embedding is None
        assert result[1].embedding == [1.0, 2.0]

    def test_truncate_empty_list(self):
        """Test truncation with empty document list."""
        result = truncate_embeddings([], 2)
        assert result == []

    def test_truncate_to_zero_dimension(self):
        """Test truncation to zero dimension."""
        docs = [
            Document(content="Doc 1", id="1", embedding=[1.0, 2.0, 3.0, 4.0]),
        ]
        result = truncate_embeddings(docs, 0)
        assert result[0].embedding == []
