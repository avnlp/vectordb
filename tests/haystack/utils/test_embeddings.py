"""Tests for EmbedderFactory utility class."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.utils.embeddings import EmbedderFactory


class TestEmbedderFactory:
    """Tests for EmbedderFactory class."""

    def test_create_document_embedder_missing_model(self) -> None:
        """Test error when model not specified."""
        with pytest.raises(KeyError):
            EmbedderFactory.create_document_embedder({})

    def test_create_text_embedder_missing_model(self) -> None:
        """Test error when model not specified."""
        with pytest.raises(KeyError):
            EmbedderFactory.create_text_embedder({})

    @patch("vectordb.haystack.utils.embeddings.SentenceTransformersDocumentEmbedder")
    def test_create_document_embedder(self, mock_embedder_class: MagicMock) -> None:
        """Test document embedder creation."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 64,
            }
        }
        result = EmbedderFactory.create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=64,
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch("vectordb.haystack.utils.embeddings.SentenceTransformersDocumentEmbedder")
    def test_create_document_embedder_with_device(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test document embedder with device specified."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {
            "embeddings": {
                "model": "test-model",
                "device": "cuda",
            }
        }
        EmbedderFactory.create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="test-model",
            batch_size=32,
            device="cuda",
        )

    @patch("vectordb.haystack.utils.embeddings.SentenceTransformersTextEmbedder")
    def test_create_text_embedder(self, mock_embedder_class: MagicMock) -> None:
        """Test text embedder creation."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "test-model"}}
        result = EmbedderFactory.create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model="test-model")
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch("vectordb.haystack.utils.embeddings.SentenceTransformersTextEmbedder")
    def test_create_text_embedder_with_device(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test text embedder with device specified."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "test-model", "device": "cpu"}}
        EmbedderFactory.create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model="test-model", device="cpu")

    def test_create_sparse_document_embedder_missing_model(self) -> None:
        """Test error when sparse model not specified."""
        with pytest.raises(KeyError):
            EmbedderFactory.create_sparse_document_embedder({})

    def test_create_sparse_text_embedder_missing_model(self) -> None:
        """Test error when sparse model not specified."""
        with pytest.raises(KeyError):
            EmbedderFactory.create_sparse_text_embedder({})

    def test_get_embedding_dimension(self) -> None:
        """Test getting embedding dimension from embedder."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {
            "documents": [MagicMock(embedding=[0.1] * 384)]
        }

        dimension = EmbedderFactory.get_embedding_dimension(mock_embedder)
        assert dimension == 384

    def test_get_embedding_dimension_no_embedding(self) -> None:
        """Test error when embedder produces no embedding."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": [MagicMock(embedding=None)]}

        with pytest.raises(ValueError, match="did not produce embeddings"):
            EmbedderFactory.get_embedding_dimension(mock_embedder)

    @patch("haystack.components.embedders.SentenceTransformersSparseDocumentEmbedder")
    def test_create_sparse_document_embedder(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test sparse document embedder creation."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"sparse": {"model": "naver/splade-cocondenser-ensembledistil"}}
        result = EmbedderFactory.create_sparse_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="naver/splade-cocondenser-ensembledistil"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch("haystack.components.embedders.SentenceTransformersSparseTextEmbedder")
    def test_create_sparse_text_embedder(self, mock_embedder_class: MagicMock) -> None:
        """Test sparse text embedder creation."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"sparse": {"model": "naver/splade-cocondenser-ensembledistil"}}
        result = EmbedderFactory.create_sparse_text_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="naver/splade-cocondenser-ensembledistil"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch("vectordb.haystack.utils.embeddings.SentenceTransformersDocumentEmbedder")
    def test_create_document_embedder_default_batch_size(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test document embedder uses default batch_size when not specified."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "test-model"}}
        EmbedderFactory.create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="test-model",
            batch_size=32,
        )
