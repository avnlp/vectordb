"""Tests for config loader module."""

import os

import pytest
import yaml
from pydantic import ValidationError

from vectordb.haystack.diversity_filtering.utils.config_loader import (
    ConfigLoader,
    DiversityFilteringConfig,
)


class TestConfigLoader:
    """Test configuration loading and validation."""

    @pytest.fixture
    def sample_config_dict(self) -> dict:
        """Sample configuration dictionary."""
        return {
            "dataset": {
                "name": "triviaqa",
                "split": "test",
                "max_documents": 1000,
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "batch_size": 32,
            },
            "index": {"name": "triviaqa_diversity"},
            "retrieval": {"top_k_candidates": 100},
            "diversity": {
                "algorithm": "maximum_margin_relevance",
                "top_k": 10,
                "mmr_lambda": 0.5,
            },
            "vectordb": {
                "type": "qdrant",
                "qdrant": {
                    "url": "http://localhost:6333",
                    "api_key": None,
                    "timeout": 30,
                },
            },
        }

    def test_load_dict_valid(self, sample_config_dict):
        """Test loading valid configuration from dict."""
        config = ConfigLoader.load_dict(sample_config_dict)

        assert isinstance(config, DiversityFilteringConfig)
        assert config.dataset.name == "triviaqa"
        assert config.dataset.split == "test"
        assert config.dataset.max_documents == 1000
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding.dimension == 384
        assert config.index.name == "triviaqa_diversity"
        assert config.retrieval.top_k_candidates == 100
        assert config.diversity.top_k == 10
        assert config.vectordb.type == "qdrant"

    def test_load_dict_missing_required(self):
        """Test loading with missing required fields."""
        invalid_config = {
            "dataset": {"name": "triviaqa"},
            # Missing index and vectordb
        }

        with pytest.raises(ValidationError):
            ConfigLoader.load_dict(invalid_config)

    def test_load_dict_invalid_dataset(self, sample_config_dict):
        """Test validation of invalid dataset name."""
        sample_config_dict["dataset"]["name"] = "invalid_dataset"

        with pytest.raises(ValidationError):
            ConfigLoader.load_dict(sample_config_dict)

    def test_load_dict_invalid_vectordb_type(self, sample_config_dict):
        """Test validation of invalid vectordb type."""
        sample_config_dict["vectordb"]["type"] = "elasticsearch"

        with pytest.raises(ValidationError):
            ConfigLoader.load_dict(sample_config_dict)

    def test_load_file_valid(self, tmp_path, sample_config_dict):
        """Test loading valid configuration from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(sample_config_dict))

        config = ConfigLoader.load(str(config_file))

        assert config.dataset.name == "triviaqa"
        assert config.index.name == "triviaqa_diversity"

    def test_load_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/path/config.yaml")

    def test_load_file_invalid_yaml(self, tmp_path):
        """Test loading malformed YAML."""
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            ConfigLoader.load(str(config_file))

    def test_env_var_substitution(self, tmp_path):
        """Test environment variable substitution in config."""
        os.environ["TEST_QDRANT_URL"] = "http://custom-qdrant:6333"
        os.environ["TEST_API_KEY"] = "test-api-key"

        config_dict = {
            "dataset": {"name": "triviaqa", "split": "test"},
            "index": {"name": "test_index"},
            "vectordb": {
                "type": "qdrant",
                "qdrant": {
                    "url": "${TEST_QDRANT_URL}",
                    "api_key": "${TEST_API_KEY}",
                },
            },
        }

        config = ConfigLoader.load_dict(config_dict)

        assert config.vectordb.qdrant.url == "http://custom-qdrant:6333"
        assert config.vectordb.qdrant.api_key == "test-api-key"

    def test_default_values(self, tmp_path):
        """Test default values for optional fields."""
        config_dict = {
            "dataset": {"name": "triviaqa"},
            "index": {"name": "test"},
            "vectordb": {
                "type": "qdrant",
                "qdrant": {"url": "http://localhost:6333"},
            },
        }

        config = ConfigLoader.load_dict(config_dict)

        # Check defaults
        assert config.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding.dimension == 384
        assert config.embedding.batch_size == 32
        assert config.retrieval.top_k_candidates == 100
        assert config.diversity.algorithm == "maximum_margin_relevance"
        assert config.diversity.top_k == 10
        assert config.rag.enabled is False

    def test_all_databases(self):
        """Test configuration for all 5 databases."""
        base_config = {
            "dataset": {"name": "triviaqa"},
            "index": {"name": "test"},
        }

        # Qdrant
        qdrant_config = base_config.copy()
        qdrant_config["vectordb"] = {
            "type": "qdrant",
            "qdrant": {"url": "http://localhost:6333"},
        }
        config = ConfigLoader.load_dict(qdrant_config)
        assert config.vectordb.type == "qdrant"

        # Pinecone
        pinecone_config = base_config.copy()
        pinecone_config["vectordb"] = {
            "type": "pinecone",
            "pinecone": {
                "api_key": "pk-test",
                "index_name": "test-index",
            },
        }
        config = ConfigLoader.load_dict(pinecone_config)
        assert config.vectordb.type == "pinecone"

        # Weaviate
        weaviate_config = base_config.copy()
        weaviate_config["vectordb"] = {
            "type": "weaviate",
            "weaviate": {"url": "http://localhost:8080"},
        }
        config = ConfigLoader.load_dict(weaviate_config)
        assert config.vectordb.type == "weaviate"

        # Chroma
        chroma_config = base_config.copy()
        chroma_config["vectordb"] = {
            "type": "chroma",
            "chroma": {"host": "localhost", "port": 8000},
        }
        config = ConfigLoader.load_dict(chroma_config)
        assert config.vectordb.type == "chroma"

        # Milvus
        milvus_config = base_config.copy()
        milvus_config["vectordb"] = {
            "type": "milvus",
            "milvus": {"host": "localhost", "port": 19530},
        }
        config = ConfigLoader.load_dict(milvus_config)
        assert config.vectordb.type == "milvus"
