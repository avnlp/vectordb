"""Tests for JSON indexing configuration utilities.

Tests cover:
- Configuration loading from YAML files
- Configuration loading from dict
- Environment variable resolution
- Default value handling
- Nested structures
"""

import os
from unittest.mock import mock_open, patch

import pytest
import yaml

from vectordb.haystack.json_indexing.common.config import (
    _resolve_env_vars,
    load_config,
)


class TestResolveEnvVars:
    """Tests for _resolve_env_vars function."""

    def test_simple_env_var(self) -> None:
        """Test resolving a simple environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _resolve_env_vars("${TEST_VAR}")
            assert result == "test_value"

    def test_env_var_with_default(self) -> None:
        """Test resolving env var with default value."""
        # Ensure env var is not set
        if "UNSET_VAR" in os.environ:
            del os.environ["UNSET_VAR"]

        result = _resolve_env_vars("${UNSET_VAR:-default_value}")
        assert result == "default_value"

    def test_env_var_with_empty_default(self) -> None:
        """Test resolving env var with empty default."""
        if "UNSET_VAR" in os.environ:
            del os.environ["UNSET_VAR"]

        result = _resolve_env_vars("${UNSET_VAR:-}")
        assert result == ""

    def test_env_var_override_default(self) -> None:
        """Test that set env var overrides default."""
        with patch.dict(os.environ, {"SET_VAR": "actual_value"}):
            result = _resolve_env_vars("${SET_VAR:-default_value}")
            assert result == "actual_value"

    def test_no_env_var_pattern(self) -> None:
        """Test string without env var pattern is returned as-is."""
        result = _resolve_env_vars("plain_string")
        assert result == "plain_string"

    def test_partial_env_var_pattern(self) -> None:
        """Test string with partial pattern is returned as-is."""
        result = _resolve_env_vars("prefix-${INCOMPLETE")
        assert result == "prefix-${INCOMPLETE"

    def test_resolve_dict(self) -> None:
        """Test resolving env vars in dictionary."""
        with patch.dict(os.environ, {"DB_HOST": "localhost", "DB_PORT": "5432"}):
            config = {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
                "name": "mydb",
            }
            result = _resolve_env_vars(config)
            assert result["host"] == "localhost"
            assert result["port"] == "5432"
            assert result["name"] == "mydb"

    def test_resolve_nested_dict(self) -> None:
        """Test resolving env vars in nested dictionary."""
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            config = {
                "database": {
                    "credentials": {
                        "api_key": "${API_KEY}",
                    },
                },
            }
            result = _resolve_env_vars(config)
            assert result["database"]["credentials"]["api_key"] == "secret123"

    def test_resolve_list(self) -> None:
        """Test resolving env vars in list."""
        with patch.dict(os.environ, {"ITEM1": "first", "ITEM2": "second"}):
            config = ["${ITEM1}", "${ITEM2}", "static"]
            result = _resolve_env_vars(config)
            assert result == ["first", "second", "static"]

    def test_resolve_mixed_structure(self) -> None:
        """Test resolving env vars in mixed structure."""
        with patch.dict(os.environ, {"HOST": "db.example.com"}):
            config = {
                "servers": ["${HOST}", "backup.example.com"],
                "ports": [5432, 5433],
                "nested": {"primary": "${HOST}"},
            }
            result = _resolve_env_vars(config)
            assert result["servers"] == ["db.example.com", "backup.example.com"]
            assert result["ports"] == [5432, 5433]
            assert result["nested"]["primary"] == "db.example.com"

    def test_resolve_non_string_values(self) -> None:
        """Test that non-string values are preserved."""
        config = {
            "count": 42,
            "enabled": True,
            "ratio": 3.14,
            "none_val": None,
        }
        result = _resolve_env_vars(config)
        assert result["count"] == 42
        assert result["enabled"] is True
        assert result["ratio"] == 3.14
        assert result["none_val"] is None

    def test_resolve_empty_structures(self) -> None:
        """Test resolving empty structures."""
        assert _resolve_env_vars({}) == {}
        assert _resolve_env_vars([]) == []
        assert _resolve_env_vars("") == ""


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_dict(self) -> None:
        """Test loading config from dictionary."""
        config_dict = {
            "database": {"host": "localhost", "port": 5432},
            "debug": True,
        }
        result = load_config(config_dict)
        assert result == config_dict

    def test_load_from_dict_with_env_vars(self) -> None:
        """Test loading config from dict with env var resolution."""
        with patch.dict(os.environ, {"DB_HOST": "remotehost"}):
            config_dict = {"host": "${DB_HOST}", "port": 5432}
            result = load_config(config_dict)
            assert result["host"] == "remotehost"
            assert result["port"] == 5432

    def test_load_from_yaml_file(self) -> None:
        """Test loading config from YAML file."""
        yaml_content = """
database:
  host: localhost
  port: 5432
debug: true
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = load_config("/path/to/config.yaml")
            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == 5432
            assert result["debug"] is True

    def test_load_from_yaml_with_env_vars(self) -> None:
        """Test loading config from YAML with env var resolution."""
        yaml_content = """
database:
  host: ${DB_HOST}
  password: ${DB_PASS:-defaultpass}
"""
        with (
            patch.dict(os.environ, {"DB_HOST": "prod.example.com"}),
            patch("builtins.open", mock_open(read_data=yaml_content)),
        ):
            result = load_config("/path/to/config.yaml")
            assert result["database"]["host"] == "prod.example.com"
            assert result["database"]["password"] == "defaultpass"

    def test_load_complex_yaml(self) -> None:
        """Test loading complex YAML configuration."""
        yaml_content = """
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  batch_size: 32

database:
  host: localhost
  port: 19530
  collection: test

search:
  top_k: 10
  filters:
    enabled: true
    fields:
      - category
      - date
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = load_config("config.yaml")
            assert (
                result["embeddings"]["model"]
                == "sentence-transformers/all-MiniLM-L6-v2"
            )
            assert result["embeddings"]["batch_size"] == 32
            assert result["database"]["collection"] == "test"
            assert result["search"]["top_k"] == 10
            assert result["search"]["filters"]["fields"] == ["category", "date"]

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml(self) -> None:
        """Test handling of invalid YAML."""
        invalid_yaml = "invalid: yaml: content: ["
        with (
            patch("builtins.open", mock_open(read_data=invalid_yaml)),
            pytest.raises(yaml.YAMLError),
        ):
            load_config("invalid.yaml")

    def test_empty_dict(self) -> None:
        """Test loading empty dictionary."""
        result = load_config({})
        assert result == {}

    def test_empty_yaml_file(self) -> None:
        """Test loading empty YAML file."""
        with patch("builtins.open", mock_open(read_data="")):
            result = load_config("empty.yaml")
            assert result is None  # yaml.safe_load returns None for empty content

    def test_yaml_with_special_characters(self) -> None:
        """Test loading YAML with special characters."""
        yaml_content = """
connection_string: 'postgresql://user:pass@host/db'
regex_pattern: '.*\\.txt$'
multiline: |
  Line 1
  Line 2
"""
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = load_config("config.yaml")
            assert "postgresql" in result["connection_string"]
            assert result["regex_pattern"] == ".*\\.txt$"
            assert "Line 1" in result["multiline"]
            assert "Line 2" in result["multiline"]
