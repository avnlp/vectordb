"""Tests for ConfigLoader configuration utility.

This module tests the ConfigLoader class which provides YAML/JSON
configuration loading, environment variable resolution, and validation
for Haystack pipeline configurations.

Key Features Tested:
    - Dictionary and YAML file configuration loading
    - Environment variable substitution with default values
    - Nested configuration resolution
    - Required section validation for database pipelines
    - Non-string value preservation (int, bool, None)

Note:
    ConfigLoader supports ${VAR} and ${VAR:-default} syntax for
    environment variable substitution in configuration values.
"""

import os
import tempfile

import pytest

from vectordb.haystack.utils import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader utility class.

    Tests cover configuration loading from dictionaries and files,
    environment variable resolution, and validation of required
    sections for database-specific pipeline configurations.
    """

    def test_load_dict(self) -> None:
        """Test loading config from dict."""
        config = {"key": "value", "nested": {"inner": "data"}}
        result = ConfigLoader.load(config)
        assert result == config

    def test_load_yaml_file(self) -> None:
        """Test loading config from YAML file."""
        config_yaml = """
dataloader:
  type: arc
  split: test
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
milvus:
  uri: http://localhost:19530
  collection_name: test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            try:
                result = ConfigLoader.load(f.name)
                assert result["dataloader"]["type"] == "arc"
                assert result["milvus"]["collection_name"] == "test"
            finally:
                os.unlink(f.name)

    def test_load_missing_file(self) -> None:
        """Test loading config from missing file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/path/config.yaml")

    def test_load_resolves_env_vars_simple(self) -> None:
        """Test that load resolves simple environment variable."""
        os.environ["TEST_VAR"] = "test_value"
        config = {"value": "${TEST_VAR}"}
        result = ConfigLoader.load(config)
        assert result["value"] == "test_value"

    def test_load_resolves_env_vars_with_default(self) -> None:
        """Test that load resolves environment variable with default."""
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]
        config = {"value": "${NONEXISTENT_VAR:-default_value}"}
        result = ConfigLoader.load(config)
        assert result["value"] == "default_value"

    def test_load_resolves_env_vars_nested_dict(self) -> None:
        """Test that load resolves environment variables in nested dict."""
        os.environ["API_KEY"] = "secret123"
        config = {
            "database": {
                "api_key": "${API_KEY}",
                "url": "http://localhost",
            }
        }
        result = ConfigLoader.load(config)
        assert result["database"]["api_key"] == "secret123"
        assert result["database"]["url"] == "http://localhost"

    def test_load_resolves_env_vars_list(self) -> None:
        """Test that load resolves environment variables in list."""
        os.environ["HOST"] = "localhost"
        config = {"hosts": ["${HOST}", "http://example.com"]}
        result = ConfigLoader.load(config)
        assert result["hosts"][0] == "localhost"
        assert result["hosts"][1] == "http://example.com"

    def test_load_preserves_non_string_values(self) -> None:
        """Test that load preserves non-string values."""
        config = {
            "number": 42,
            "boolean": True,
            "none": None,
        }
        result = ConfigLoader.load(config)
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["none"] is None

    def test_validate_missing_dataloader(self) -> None:
        """Test validation fails without dataloader."""
        config = {
            "embeddings": {"model": "test"},
            "milvus": {"uri": "http://localhost"},
        }
        with pytest.raises(ValueError, match="dataloader"):
            ConfigLoader.validate(config, "milvus")

    def test_validate_missing_embeddings(self) -> None:
        """Test validation fails without embeddings."""
        config = {
            "dataloader": {"type": "arc"},
            "milvus": {"uri": "http://localhost"},
        }
        with pytest.raises(ValueError, match="embeddings"):
            ConfigLoader.validate(config, "milvus")

    def test_validate_missing_db_config(self) -> None:
        """Test validation fails without DB config."""
        config = {
            "dataloader": {"type": "arc"},
            "embeddings": {"model": "test"},
        }
        with pytest.raises(ValueError, match="milvus"):
            ConfigLoader.validate(config, "milvus")

    def test_validate_success(self, milvus_config: dict) -> None:
        """Test validation succeeds with all required keys."""
        ConfigLoader.validate(milvus_config, "milvus")
