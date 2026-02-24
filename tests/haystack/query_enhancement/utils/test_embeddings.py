"""Tests for query enhancement embeddings utilities."""

from typing import Any
from unittest.mock import MagicMock, patch

from vectordb.haystack.query_enhancement.utils.embeddings import (
    MODEL_ALIASES,
    create_document_embedder,
    create_text_embedder,
)


class TestCreateTextEmbedder:
    """Tests for create_text_embedder function."""

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_default_config(self, mock_embedder_class: MagicMock) -> None:
        """Test with default config (should use minilm model)."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config: dict[str, Any] = {}
        result = create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_custom_model_in_config(self, mock_embedder_class: MagicMock) -> None:
        """Test with custom model in config['embeddings']['model']."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "custom/my-embedding-model"}}
        result = create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model="custom/my-embedding-model")
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_model_alias_qwen3(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution for 'qwen3'."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "qwen3"}}
        create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["qwen3"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_model_alias_minilm(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution for 'minilm'."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "minilm"}}
        create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["minilm"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_model_alias_mpnet(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution for 'mpnet'."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "mpnet"}}
        create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["mpnet"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_model_alias_case_insensitive(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution is case insensitive."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "QWEN3"}}
        create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["qwen3"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_warmup_is_called(self, mock_embedder_class: MagicMock) -> None:
        """Test that warmup is called on the embedder."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        create_text_embedder({})

        mock_instance.warm_up.assert_called_once()


class TestCreateDocumentEmbedder:
    """Tests for create_document_embedder function."""

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_default_config(self, mock_embedder_class: MagicMock) -> None:
        """Test with default config (should use minilm model)."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config: dict[str, Any] = {}
        result = create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_custom_model_in_config(self, mock_embedder_class: MagicMock) -> None:
        """Test with custom model in config['embeddings']['model']."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "custom/my-doc-embedding-model"}}
        result = create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="custom/my-doc-embedding-model"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_model_alias_qwen3(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution for 'qwen3'."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "qwen3"}}
        create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["qwen3"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_model_alias_minilm(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution for 'minilm'."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "minilm"}}
        create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["minilm"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_model_alias_mpnet(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution for 'mpnet'."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "mpnet"}}
        create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["mpnet"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_model_alias_case_insensitive(self, mock_embedder_class: MagicMock) -> None:
        """Test model alias resolution is case insensitive."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "MiniLM"}}
        create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(model=MODEL_ALIASES["minilm"])

    @patch(
        "vectordb.haystack.query_enhancement.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_warmup_is_called(self, mock_embedder_class: MagicMock) -> None:
        """Test that warmup is called on the embedder."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        create_document_embedder({})

        mock_instance.warm_up.assert_called_once()


class TestModelAliases:
    """Tests for MODEL_ALIASES constant."""

    def test_aliases_exist(self) -> None:
        """Test that expected model aliases exist."""
        assert "qwen3" in MODEL_ALIASES
        assert "minilm" in MODEL_ALIASES
        assert "mpnet" in MODEL_ALIASES

    def test_alias_values(self) -> None:
        """Test that aliases map to expected model names."""
        assert MODEL_ALIASES["qwen3"] == "Qwen/Qwen3-Embedding-0.6B"
        assert MODEL_ALIASES["minilm"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert MODEL_ALIASES["mpnet"] == "sentence-transformers/all-mpnet-base-v2"
