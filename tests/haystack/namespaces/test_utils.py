"""Tests for namespace utilities."""

from unittest.mock import Mock, patch

import pytest
import yaml

from vectordb.haystack.namespaces.utils import (
    Timer,
    get_document_embedder,
    get_text_embedder,
    load_config,
)


class TestUtils:
    """Test suite for namespace utilities."""

    def test_load_config(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "pipeline": {"name": "test-pipeline"},
            "embedding": {"model": "test-model"},
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config["pipeline"]["name"] == "test-pipeline"
        assert config["embedding"]["model"] == "test-model"

    def test_load_config_nonexistent(self):
        """Test loading nonexistent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_timer_context_manager(self):
        """Test Timer context manager functionality."""
        with Timer() as timer:
            timer._start = timer._start

        # Timer should have recorded some time (may be very small)
        assert isinstance(timer.elapsed_ms, float)
        assert timer.elapsed_ms >= 0.0

    @patch(
        "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_get_document_embedder(self, mock_embedder_class, tmp_path):
        """Test getting document embedder from config."""
        config_path = tmp_path / "config.yaml"
        config_data = {"embedding": {"model": "Qwen/Qwen3-Embedding-0.6B"}}
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        mock_embedder_instance = Mock()
        mock_embedder_class.return_value = mock_embedder_instance

        config = load_config(str(config_path))
        embedder = get_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )
        mock_embedder_instance.warm_up.assert_called_once()
        assert embedder is mock_embedder_instance

    @patch(
        "vectordb.haystack.namespaces.utils.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_get_text_embedder(self, mock_embedder_class, tmp_path):
        """Test getting text embedder from config."""
        config_path = tmp_path / "config.yaml"
        config_data = {"embedding": {"model": "Qwen/Qwen3-Embedding-0.6B"}}
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        mock_embedder_instance = Mock()
        mock_embedder_class.return_value = mock_embedder_instance

        config = load_config(str(config_path))
        embedder = get_text_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )
        mock_embedder_instance.warm_up.assert_called_once()
        assert embedder is mock_embedder_instance
