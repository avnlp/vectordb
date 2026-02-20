"""Tests for ConfigLoader utility class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vectordb.haystack.utils.config import ConfigLoader


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_from_dict(self) -> None:
        """Test loading config from dictionary."""
        config = {"key": "value", "nested": {"inner": 123}}
        result = ConfigLoader.load(config)
        assert result == config

    def test_load_from_yaml_file(self) -> None:
        """Test loading config from YAML file."""
        yaml_content = """
dataloader:
  type: arc
  split: test
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            result = ConfigLoader.load(config_path)
            assert result["dataloader"]["type"] == "arc"
            assert (
                result["embeddings"]["model"]
                == "sentence-transformers/all-MiniLM-L6-v2"
            )
        finally:
            config_path.unlink()

    def test_env_var_substitution_simple(self) -> None:
        """Test environment variable substitution."""
        with patch.dict(os.environ, {"MY_VAR": "my_value"}):
            config = {"key": "${MY_VAR}"}
            result = ConfigLoader.load(config)
            assert result["key"] == "my_value"

    def test_env_var_substitution_with_default(self) -> None:
        """Test environment variable with default value."""
        os.environ.pop("MISSING_VAR", None)
        config = {"key": "${MISSING_VAR:-default_value}"}
        result = ConfigLoader.load(config)
        assert result["key"] == "default_value"

    def test_env_var_substitution_nested(self) -> None:
        """Test environment variable substitution in nested config."""
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            config = {
                "database": {
                    "api_key": "${API_KEY}",
                    "host": "localhost",
                }
            }
            result = ConfigLoader.load(config)
            assert result["database"]["api_key"] == "secret123"
            assert result["database"]["host"] == "localhost"

    def test_env_var_substitution_in_list(self) -> None:
        """Test environment variable substitution in lists."""
        with patch.dict(os.environ, {"ITEM": "value"}):
            config = {"items": ["${ITEM}", "static"]}
            result = ConfigLoader.load(config)
            assert result["items"] == ["value", "static"]

    def test_validate_success(self) -> None:
        """Test validation passes with required sections."""
        config = {
            "dataloader": {"type": "arc"},
            "embeddings": {"model": "test"},
            "pinecone": {"api_key": "key"},
        }
        ConfigLoader.validate(config, "pinecone")

    def test_validate_missing_sections(self) -> None:
        """Test validation fails with missing sections."""
        config = {"dataloader": {"type": "arc"}}
        with pytest.raises(ValueError, match="Missing required config sections"):
            ConfigLoader.validate(config, "pinecone")

    def test_non_string_values_unchanged(self) -> None:
        """Test non-string values pass through unchanged."""
        config = {
            "count": 42,
            "enabled": True,
            "ratio": 0.5,
            "items": [1, 2, 3],
        }
        result = ConfigLoader.load(config)
        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["ratio"] == 0.5
        assert result["items"] == [1, 2, 3]
