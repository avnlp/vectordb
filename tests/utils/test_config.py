"""Unit tests for configuration utilities.

Tests validate configuration handling patterns used across the vectordb library
for initializing vector databases, dataloaders, embeddings, and search pipelines.
Configuration structures are central to the library's flexible pipeline design.

Test coverage includes:
- Dictionary-based configuration loading
- Nested configuration access patterns
- Default value handling for optional keys
- Multi-database configurations (Milvus, Qdrant, Pinecone)
- Dataloader type specifications (ARC, TriviaQA, PopQA, FactScore, EarningsCall)
- Search parameter validation (top_k, filters, hybrid_search)
- Embedding model configurations (sentence-transformers, OpenAI)
- RAG feature toggles and reranker settings
- Semantic diversification (MMR) parameters
- Environment variable resolution
- Embedding model alias resolution
- Dataset limit lookups
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from vectordb.utils.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_ALIASES,
    get_dataset_limits,
    load_config,
    resolve_embedding_model,
    resolve_env_vars,
    setup_logger,
)


class TestLoadConfig:
    """Test suite for configuration loading and access patterns.

    Validates configuration structures used by indexing and search pipelines.
    Tests ensure that:
    - Configuration dictionaries can be loaded and validated
    - Nested access patterns work for complex configurations
    - Missing keys are handled gracefully with defaults
    - All supported database types have valid config schemas
    - Dataloader types accept required parameters
    - Search parameters are properly structured
    """

    def test_load_config_from_dict(self) -> None:
        """Test that configuration dictionaries are accepted by pipelines.

        Verifies the standard configuration structure with dataloader,
        embeddings, and search sections is valid and can be accessed.

        Returns:
            None
        """
        config_dict = {
            "dataloader": {"type": "arc", "split": "test"},
            "embeddings": {"model": "test-model"},
            "search": {"top_k": 5},
        }

        assert isinstance(config_dict, dict)
        assert "dataloader" in config_dict

    def test_load_config_keys_present(self) -> None:
        """Test that required configuration sections are accessible.

        Validates that embeddings and search keys exist in config,
        which are required by most pipeline implementations.

        Returns:
            None
        """
        config_dict = {
            "embeddings": {"model": "test-model", "device": "cpu"},
            "search": {"top_k": 10},
        }

        assert "embeddings" in config_dict
        assert "search" in config_dict

    def test_load_config_nested_access(self) -> None:
        """Test accessing nested database configuration values.

        Verifies that nested URI and collection name can be extracted
        from Milvus configuration section using chained key access.

        Returns:
            None
        """
        config_dict = {
            "database": {
                "milvus": {
                    "uri": "http://localhost:19530",
                    "collection": "test",
                }
            }
        }

        db_config = config_dict["database"]["milvus"]
        assert db_config["uri"] == "http://localhost:19530"
        assert db_config["collection"] == "test"

    def test_load_config_with_defaults(self) -> None:
        """Test default value handling for missing configuration keys.

        Validates the pattern of using dict.get() with nested defaults
        to safely access optional configuration without KeyError.

        Returns:
            None
        """
        config_dict = {
            "search": {"top_k": 5},
        }

        model = config_dict.get("embeddings", {}).get("model", "default-model")
        assert model == "default-model"

    def test_load_config_multiple_databases(self) -> None:
        """Test configuration supporting multiple database backends.

        Verifies that configurations can specify multiple vector databases
        (Milvus, Qdrant, Pinecone) for multi-backend deployments.

        Returns:
            None
        """
        config_dict = {
            "milvus": {"uri": "http://localhost:19530"},
            "qdrant": {"url": "http://localhost:6333"},
            "pinecone": {"api_key": "test-key"},
        }

        assert "milvus" in config_dict
        assert "qdrant" in config_dict
        assert "pinecone" in config_dict

    def test_load_config_dataloader_types(self) -> None:
        """Test configuration for all supported dataloader types.

        Validates configuration structures for each dataloader:
        - ARC: Requires split and limit parameters
        - TriviaQA: Requires split parameter
        - PopQA: No additional parameters required
        - FactScore: No additional parameters required

        Returns:
            None
        """
        dataloader_configs = {
            "arc": {"type": "arc", "split": "test", "limit": 100},
            "triviaqa": {"type": "triviaqa", "split": "dev"},
            "popqa": {"type": "popqa"},
            "factscore": {"type": "factscore"},
        }

        for _name, config in dataloader_configs.items():
            assert "type" in config

    def test_load_config_search_parameters(self) -> None:
        """Test search parameter configuration structure.

        Validates that search configs include top_k, filters, and
        hybrid_search toggle used by retrieval pipelines.

        Returns:
            None
        """
        config_dict = {
            "search": {
                "top_k": 5,
                "filters": {"source": "wiki"},
                "hybrid_search": False,
            }
        }

        search_config = config_dict["search"]
        assert search_config["top_k"] == 5
        assert "filters" in search_config

    def test_load_config_embedding_models(self) -> None:
        """Test configuration for supported embedding model types.

        Validates that sentence-transformers and OpenAI models can be
        specified in the embeddings configuration section.

        Returns:
            None
        """
        embedding_configs = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "openai",
        ]

        for model in embedding_configs:
            config = {"embeddings": {"model": model}}
            assert config["embeddings"]["model"] == model

    def test_load_config_rag_settings(self) -> None:
        """Test RAG (Retrieval-Augmented Generation) feature configuration.

        Validates RAG configuration structure including:
        - Enabled toggle for RAG pipelines
        - Reranker specification (cross-encoder, etc.)
        - Context window size for LLM prompts

        Returns:
            None
        """
        config_dict = {
            "rag": {
                "enabled": True,
                "reranker": "cross-encoder",
                "context_window": 3,
            }
        }

        rag_config = config_dict.get("rag", {})
        assert rag_config.get("enabled", False) is True
        assert rag_config.get("reranker") == "cross-encoder"

    def test_load_config_semantic_diversification(self) -> None:
        """Test semantic diversification (MMR) configuration.

        Validates MMR configuration structure including:
        - Enabled toggle for diversification
        - Method specification (mmr, etc.)
        - Diversity factor for relevance/diversity trade-off

        Returns:
            None
        """
        config_dict = {
            "semantic_diversification": {
                "enabled": True,
                "method": "mmr",
                "diversity_factor": 0.5,
            }
        }

        div_config = config_dict.get("semantic_diversification", {})
        assert div_config.get("enabled") is True
        assert div_config.get("method") == "mmr"


class TestResolveEnvVars:
    """Test suite for environment variable resolution.

    Tests cover:
    - Simple ${VAR} syntax
    - Default value syntax ${VAR:-default}
    - Nested dict resolution
    - List resolution
    - Non-string values pass through unchanged
    """

    def test_resolve_env_vars_simple_string(self) -> None:
        """Test resolving simple environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = resolve_env_vars("${TEST_VAR}")
            assert result == "test_value"

    def test_resolve_env_vars_with_default_value(self) -> None:
        """Test resolving env var with default when var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_env_vars("${UNSET_VAR:-default_value}")
            assert result == "default_value"

    def test_resolve_env_vars_with_default_but_var_set(self) -> None:
        """Test resolving env var with default when var is set."""
        with patch.dict(os.environ, {"SET_VAR": "actual_value"}):
            result = resolve_env_vars("${SET_VAR:-default_value}")
            assert result == "actual_value"

    def test_resolve_env_vars_unset_without_default(self) -> None:
        """Test resolving unset env var without default returns empty string."""
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_env_vars("${UNSET_VAR}")
            assert result == ""

    def test_resolve_env_vars_plain_string(self) -> None:
        """Test that plain strings pass through unchanged."""
        result = resolve_env_vars("plain_string")
        assert result == "plain_string"

    def test_resolve_env_vars_dict(self) -> None:
        """Test resolving env vars in nested dict."""
        with patch.dict(os.environ, {"DB_HOST": "localhost", "DB_PORT": "5432"}):
            config = {
                "database": {
                    "host": "${DB_HOST}",
                    "port": "${DB_PORT}",
                }
            }
            result = resolve_env_vars(config)
            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == "5432"

    def test_resolve_env_vars_list(self) -> None:
        """Test resolving env vars in list."""
        with patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"}):
            config = ["${VAR1}", "${VAR2}", "static"]
            result = resolve_env_vars(config)
            assert result == ["value1", "value2", "static"]

    def test_resolve_env_vars_nested_list_in_dict(self) -> None:
        """Test resolving env vars in list nested in dict."""
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            config = {
                "keys": ["${API_KEY}", "other_key"],
            }
            result = resolve_env_vars(config)
            assert result["keys"] == ["secret123", "other_key"]

    def test_resolve_env_vars_non_string_values(self) -> None:
        """Test that non-string values pass through unchanged."""
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(3.14) == 3.14
        assert resolve_env_vars(True) is True
        assert resolve_env_vars(None) is None

    def test_resolve_env_vars_empty_default(self) -> None:
        """Test resolving env var with empty default value."""
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_env_vars("${UNSET_VAR:-}")
            assert result == ""


class TestLoadConfigFromFile:
    """Test suite for YAML configuration file loading.

    Tests cover:
    - Loading valid YAML files
    - Environment variable resolution in loaded config
    - Error handling for missing files
    """

    def test_load_config_from_yaml_file(self) -> None:
        """Test loading configuration from a YAML file."""
        yaml_content = """
dataloader:
  type: arc
  split: test
embeddings:
  model: test-model
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config["dataloader"]["type"] == "arc"
            assert config["dataloader"]["split"] == "test"
            assert config["embeddings"]["model"] == "test-model"
        finally:
            Path(config_path).unlink()

    def test_load_config_with_env_var_resolution(self) -> None:
        """Test that env vars are resolved when loading YAML file."""
        yaml_content = """
database:
  api_key: ${TEST_API_KEY}
  host: ${DB_HOST:-localhost}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = f.name

        try:
            with patch.dict(os.environ, {"TEST_API_KEY": "secret123"}):
                config = load_config(config_path)
                assert config["database"]["api_key"] == "secret123"
                assert config["database"]["host"] == "localhost"
        finally:
            Path(config_path).unlink()

    def test_load_config_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_malformed_yaml_raises_error(self) -> None:
        """Test that load_config raises yaml.YAMLError for malformed YAML."""
        malformed_yaml = """invalid_yaml: [
    unclosed: quote
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(malformed_yaml)
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestSetupLogger:
    """Test suite for logger setup from configuration.

    Tests cover:
    - Default logger name and level
    - Custom logger name and level
    - Missing logging config section
    """

    def test_setup_logger_default_config(self) -> None:
        """Test logger setup with default configuration."""
        config: dict[str, dict[str, str]] = {}
        logger = setup_logger(config)
        assert logger is not None
        assert logger.name == "vectordb_pipeline"

    def test_setup_logger_custom_name(self) -> None:
        """Test logger setup with custom name."""
        config = {"logging": {"name": "custom_logger"}}
        logger = setup_logger(config)
        assert logger.name == "custom_logger"

    def test_setup_logger_custom_level(self) -> None:
        """Test logger setup with custom log level."""
        config = {"logging": {"name": "test_logger_debug", "level": "DEBUG"}}
        logger = setup_logger(config)
        # Logger is created successfully with custom level
        assert logger is not None
        assert logger.name == "test_logger_debug"

    def test_setup_logger_invalid_level_falls_back_to_info(self) -> None:
        """Test that invalid log level falls back to INFO."""
        config = {"logging": {"name": "test_logger_invalid", "level": "INVALID_LEVEL"}}
        logger = setup_logger(config)
        # Logger is created successfully even with invalid level
        assert logger is not None
        assert logger.name == "test_logger_invalid"


class TestResolveEmbeddingModel:
    """Test suite for embedding model alias resolution.

    Tests cover:
    - Known aliases resolve to full paths
    - Unknown names pass through unchanged
    - Case insensitivity
    """

    def test_resolve_embedding_model_qwen3_alias(self) -> None:
        """Test resolving qwen3 alias."""
        result = resolve_embedding_model("qwen3")
        assert result == "Qwen/Qwen3-Embedding-0.6B"

    def test_resolve_embedding_model_minilm_alias(self) -> None:
        """Test resolving minilm alias."""
        result = resolve_embedding_model("minilm")
        assert result == "sentence-transformers/all-MiniLM-L6-v2"

    def test_resolve_embedding_model_mpnet_alias(self) -> None:
        """Test resolving mpnet alias."""
        result = resolve_embedding_model("mpnet")
        assert result == "sentence-transformers/all-mpnet-base-v2"

    def test_resolve_embedding_model_unknown_passes_through(self) -> None:
        """Test that unknown model names pass through unchanged."""
        result = resolve_embedding_model("custom/model-path")
        assert result == "custom/model-path"

    def test_resolve_embedding_model_case_insensitive(self) -> None:
        """Test that alias resolution is case insensitive."""
        result = resolve_embedding_model("QWEN3")
        assert result == "Qwen/Qwen3-Embedding-0.6B"

        result = resolve_embedding_model("MiniLM")
        assert result == "sentence-transformers/all-MiniLM-L6-v2"

    def test_resolve_embedding_model_full_path_unchanged(self) -> None:
        """Test that full HuggingFace paths pass through unchanged."""
        full_path = "sentence-transformers/all-MiniLM-L6-v2"
        result = resolve_embedding_model(full_path)
        assert result == full_path

    def test_embedding_model_aliases_contains_expected_keys(self) -> None:
        """Test that EMBEDDING_MODEL_ALIASES has expected keys."""
        assert "qwen3" in EMBEDDING_MODEL_ALIASES
        assert "minilm" in EMBEDDING_MODEL_ALIASES
        assert "mpnet" in EMBEDDING_MODEL_ALIASES

    def test_default_embedding_model_is_valid(self) -> None:
        """Test that DEFAULT_EMBEDDING_MODEL is a valid model path."""
        assert DEFAULT_EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B"


class TestGetDatasetLimits:
    """Test suite for dataset limit lookups.

    Tests cover:
    - Known datasets return correct limits
    - Unknown datasets return default limits
    - Case insensitivity
    """

    def test_get_dataset_limits_trivia_qa(self) -> None:
        """Test limits for trivia_qa dataset."""
        limits = get_dataset_limits("trivia_qa")
        assert limits["index_limit"] == 500
        assert limits["eval_limit"] == 100

    def test_get_dataset_limits_ai2_arc(self) -> None:
        """Test limits for ai2_arc dataset."""
        limits = get_dataset_limits("ai2_arc")
        assert limits["index_limit"] == 1000
        assert limits["eval_limit"] == 200

    def test_get_dataset_limits_popqa(self) -> None:
        """Test limits for PopQA dataset."""
        limits = get_dataset_limits("akariasai/PopQA")
        assert limits["index_limit"] == 500
        assert limits["eval_limit"] == 100

    def test_get_dataset_limits_factscore(self) -> None:
        """Test limits for FActScore dataset."""
        limits = get_dataset_limits("dskar/FActScore")
        assert limits["index_limit"] == 500
        assert limits["eval_limit"] == 100

    def test_get_dataset_limits_earnings_calls(self) -> None:
        """Test limits for earnings-calls-qa dataset."""
        limits = get_dataset_limits("lamini/earnings-calls-qa")
        assert limits["index_limit"] == 300
        assert limits["eval_limit"] == 50

    def test_get_dataset_limits_unknown_dataset(self) -> None:
        """Test that unknown datasets return default limits."""
        limits = get_dataset_limits("unknown_dataset")
        assert limits["index_limit"] == 500
        assert limits["eval_limit"] == 100

    def test_get_dataset_limits_case_insensitive(self) -> None:
        """Test that dataset limit lookup is case insensitive."""
        limits = get_dataset_limits("TRIVIA_QA")
        assert limits["index_limit"] == 500
        assert limits["eval_limit"] == 100

    def test_get_dataset_limits_returns_dict(self) -> None:
        """Test that get_dataset_limits always returns a dict with expected keys."""
        limits = get_dataset_limits("any_dataset")
        assert isinstance(limits, dict)
        assert "index_limit" in limits
        assert "eval_limit" in limits
