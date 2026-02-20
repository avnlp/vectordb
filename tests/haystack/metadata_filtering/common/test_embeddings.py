"""Tests for embeddings utility functions."""

from unittest.mock import MagicMock, patch


class TestGetDocumentEmbedder:
    """Tests for get_document_embedder function."""

    def test_get_document_embedder_default_model(self):
        """Test document embedder with default model."""
        from vectordb.haystack.metadata_filtering.common.embeddings import (
            get_document_embedder,
        )

        config = {}
        with patch(
            "vectordb.haystack.metadata_filtering.common.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_document_embedder(config)

            mock_class.assert_called_once_with(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance

    def test_get_document_embedder_custom_model(self):
        """Test document embedder with custom model."""
        from vectordb.haystack.metadata_filtering.common.embeddings import (
            get_document_embedder,
        )

        config = {"embeddings": {"model": "custom-model/all-MiniLM-L6-v2"}}
        with patch(
            "vectordb.haystack.metadata_filtering.common.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_document_embedder(config)

            mock_class.assert_called_once_with(model="custom-model/all-MiniLM-L6-v2")
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance

    def test_get_document_embedder_empty_config(self):
        """Test document embedder with empty config uses default."""
        from vectordb.haystack.metadata_filtering.common.embeddings import (
            get_document_embedder,
        )

        config = {"embeddings": {}}
        with patch(
            "vectordb.haystack.metadata_filtering.common.embeddings.SentenceTransformersDocumentEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_document_embedder(config)

            mock_class.assert_called_once_with(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance


class TestGetTextEmbedder:
    """Tests for get_text_embedder function."""

    def test_get_text_embedder_default_model(self):
        """Test text embedder with default model."""
        from vectordb.haystack.metadata_filtering.common.embeddings import (
            get_text_embedder,
        )

        config = {}
        with patch(
            "vectordb.haystack.metadata_filtering.common.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_text_embedder(config)

            mock_class.assert_called_once_with(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance

    def test_get_text_embedder_custom_model(self):
        """Test text embedder with custom model."""
        from vectordb.haystack.metadata_filtering.common.embeddings import (
            get_text_embedder,
        )

        config = {"embeddings": {"model": "custom-model/e5-small-v2"}}
        with patch(
            "vectordb.haystack.metadata_filtering.common.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_text_embedder(config)

            mock_class.assert_called_once_with(model="custom-model/e5-small-v2")
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance

    def test_get_text_embedder_no_embeddings_config(self):
        """Test text embedder when embeddings config is missing."""
        from vectordb.haystack.metadata_filtering.common.embeddings import (
            get_text_embedder,
        )

        config = {}
        with patch(
            "vectordb.haystack.metadata_filtering.common.embeddings.SentenceTransformersTextEmbedder"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = get_text_embedder(config)

            mock_class.assert_called_once_with(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            mock_instance.warm_up.assert_called_once()
            assert result == mock_instance
