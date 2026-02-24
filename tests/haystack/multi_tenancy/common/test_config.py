"""Tests for multi-tenancy configuration utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from vectordb.haystack.multi_tenancy.common.config import (
    get_database_type,
    load_config,
    resolve_env_vars,
)


class TestResolveEnvVars:
    """Tests for resolve_env_vars function."""

    def test_resolve_env_vars_string_no_vars(self):
        """Test resolving string with no environment variables."""
        result = resolve_env_vars("simple string")
        assert result == "simple string"

    def test_resolve_env_vars_integer(self):
        """Test resolving integer value."""
        result = resolve_env_vars(42)
        assert result == 42

    def test_resolve_env_vars_boolean(self):
        """Test resolving boolean value."""
        result = resolve_env_vars(True)
        assert result is True

    def test_resolve_env_vars_list(self):
        """Test resolving list with mixed types."""
        result = resolve_env_vars(["string", 42, True])
        assert result == ["string", 42, True]

    def test_resolve_env_vars_dict(self):
        """Test resolving dictionary with mixed values."""
        result = resolve_env_vars({"key": "value", "number": 123})
        assert result == {"key": "value", "number": 123}

    def test_resolve_env_vars_with_env_var(self):
        """Test resolving string with environment variable."""
        os.environ["TEST_VAR"] = "resolved_value"
        try:
            result = resolve_env_vars("${TEST_VAR}")
            assert result == "resolved_value"
        finally:
            del os.environ["TEST_VAR"]

    def test_resolve_env_vars_with_default(self):
        """Test resolving string with environment variable and default."""
        result = resolve_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_resolve_env_vars_nested_dict(self):
        """Test resolving nested dictionary with env vars."""
        os.environ["NESTED_VAR"] = "nested_value"
        try:
            result = resolve_env_vars({"outer": {"inner": "${NESTED_VAR}"}})
            assert result == {"outer": {"inner": "nested_value"}}
        finally:
            del os.environ["NESTED_VAR"]

    def test_resolve_env_vars_nested_list(self):
        """Test resolving nested list with env vars."""
        os.environ["LIST_VAR"] = "list_value"
        try:
            result = resolve_env_vars([["${LIST_VAR}"], ["other"]])
            assert result == [["list_value"], ["other"]]
        finally:
            del os.environ["LIST_VAR"]

    def test_resolve_env_vars_none(self):
        """Test resolving None value."""
        result = resolve_env_vars(None)
        assert result is None


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        config_dict = {"key": "value", "nested": {"inner": True}}
        result = load_config(config_dict)
        assert result == config_dict

    def test_load_config_from_file(self):
        """Test loading config from YAML file."""
        yaml_content = """
key: value
nested:
  inner: true
number: 42
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = load_config(f.name)
                assert result["key"] == "value"
                assert result["nested"]["inner"] is True
                assert result["number"] == 42
            finally:
                os.unlink(f.name)

    def test_load_config_from_path_object(self):
        """Test loading config from Path object."""
        yaml_content = """
database:
  type: milvus
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = load_config(Path(f.name))
                assert result["database"]["type"] == "milvus"
            finally:
                os.unlink(f.name)

    def test_load_config_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_empty_file_returns_none(self):
        """Test that empty YAML file returns None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            try:
                result = load_config(f.name)
                assert result is None
            finally:
                os.unlink(f.name)

    def test_load_config_resolves_env_vars(self):
        """Test that loading config resolves environment variables."""
        os.environ["CONFIG_VAR"] = "env_resolved_value"
        yaml_content = """
key: ${CONFIG_VAR}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = load_config(f.name)
                assert result["key"] == "env_resolved_value"
            finally:
                os.unlink(f.name)
                del os.environ["CONFIG_VAR"]

    def test_load_config_with_dict_resolves_env_vars(self):
        """Test that dict input also resolves environment variables."""
        os.environ["DICT_VAR"] = "dict_value"
        try:
            config_dict = {"key": "${DICT_VAR}"}
            result = load_config(config_dict)
            assert result["key"] == "dict_value"
        finally:
            del os.environ["DICT_VAR"]


class TestGetDatabaseType:
    """Tests for get_database_type function."""

    def test_get_database_type_milvus(self):
        """Test getting database type for Milvus."""
        config = {"database": {"type": "milvus"}}
        result = get_database_type(config)
        assert result == "milvus"

    def test_get_database_type_weaviate(self):
        """Test getting database type for Weaviate."""
        config = {"database": {"type": "weaviate"}}
        result = get_database_type(config)
        assert result == "weaviate"

    def test_get_database_type_pinecone(self):
        """Test getting database type for Pinecone."""
        config = {"database": {"type": "pinecone"}}
        result = get_database_type(config)
        assert result == "pinecone"

    def test_get_database_type_qdrant(self):
        """Test getting database type for Qdrant."""
        config = {"database": {"type": "qdrant"}}
        result = get_database_type(config)
        assert result == "qdrant"

    def test_get_database_type_chroma(self):
        """Test getting database type for Chroma."""
        config = {"database": {"type": "chroma"}}
        result = get_database_type(config)
        assert result == "chroma"

    def test_get_database_type_default(self):
        """Test default database type when not specified."""
        config = {}
        result = get_database_type(config)
        assert result == "milvus"

    def test_get_database_type_uppercase(self):
        """Test that database type is converted to lowercase."""
        config = {"database": {"type": "MILVUS"}}
        result = get_database_type(config)
        assert result == "milvus"
