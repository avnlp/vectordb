"""Tests for configuration loading and validation utilities."""

import os
import tempfile

import pytest


class TestResolveEnvVars:
    """Tests for resolve_env_vars function."""

    def test_resolve_env_vars_string_no_vars(self):
        """Test resolving string with no environment variables."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        result = resolve_env_vars("simple string")
        assert result == "simple string"

    def test_resolve_env_vars_integer(self):
        """Test resolving integer value."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        result = resolve_env_vars(42)
        assert result == 42

    def test_resolve_env_vars_boolean(self):
        """Test resolving boolean value."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        result = resolve_env_vars(True)
        assert result is True

    def test_resolve_env_vars_list(self):
        """Test resolving list with mixed types."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        result = resolve_env_vars(["string", 42, True])
        assert result == ["string", 42, True]

    def test_resolve_env_vars_dict(self):
        """Test resolving dictionary with mixed values."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        result = resolve_env_vars({"key": "value", "number": 123})
        assert result == {"key": "value", "number": 123}

    def test_resolve_env_vars_with_env_var(self):
        """Test resolving string with environment variable."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        os.environ["TEST_VAR"] = "resolved_value"
        try:
            result = resolve_env_vars("${TEST_VAR}")
            assert result == "resolved_value"
        finally:
            del os.environ["TEST_VAR"]

    def test_resolve_env_vars_with_default(self):
        """Test resolving string with environment variable and default."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        result = resolve_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_resolve_env_vars_nested_dict(self):
        """Test resolving nested dictionary with env vars."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        os.environ["NESTED_VAR"] = "nested_value"
        try:
            result = resolve_env_vars({"outer": {"inner": "${NESTED_VAR}"}})
            assert result == {"outer": {"inner": "nested_value"}}
        finally:
            del os.environ["NESTED_VAR"]

    def test_resolve_env_vars_nested_list(self):
        """Test resolving nested list with env vars."""
        from vectordb.haystack.metadata_filtering.common.config import resolve_env_vars

        os.environ["LIST_VAR"] = "list_value"
        try:
            result = resolve_env_vars([["${LIST_VAR}"], ["other"]])
            assert result == [["list_value"], ["other"]]
        finally:
            del os.environ["LIST_VAR"]


class TestLoadMetadataFilteringConfig:
    """Tests for load_metadata_filtering_config function."""

    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        from vectordb.haystack.metadata_filtering.common.config import (
            load_metadata_filtering_config,
        )

        config_dict = {"key": "value", "nested": {"inner": True}}
        result = load_metadata_filtering_config(config_dict)
        assert result == config_dict

    def test_load_config_from_file(self):
        """Test loading config from YAML file."""
        from vectordb.haystack.metadata_filtering.common.config import (
            load_metadata_filtering_config,
        )

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
                result = load_metadata_filtering_config(f.name)
                assert result["key"] == "value"
                assert result["nested"]["inner"] is True
                assert result["number"] == 42
            finally:
                os.unlink(f.name)

    def test_load_config_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        from vectordb.haystack.metadata_filtering.common.config import (
            load_metadata_filtering_config,
        )

        with pytest.raises(FileNotFoundError):
            load_metadata_filtering_config("/nonexistent/path/config.yaml")

    def test_load_config_empty_file_returns_empty_dict(self):
        """Test that empty YAML file returns empty dict."""
        from vectordb.haystack.metadata_filtering.common.config import (
            load_metadata_filtering_config,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            try:
                result = load_metadata_filtering_config(f.name)
                assert result == {}
            finally:
                os.unlink(f.name)

    def test_load_config_resolves_env_vars(self):
        """Test that loading config resolves environment variables."""
        from vectordb.haystack.metadata_filtering.common.config import (
            load_metadata_filtering_config,
        )

        os.environ["CONFIG_VAR"] = "env_resolved_value"
        yaml_content = """
key: ${CONFIG_VAR}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                result = load_metadata_filtering_config(f.name)
                assert result["key"] == "env_resolved_value"
            finally:
                os.unlink(f.name)
                del os.environ["CONFIG_VAR"]

    def test_load_config_none_input_returns_empty_dict(self):
        """Test that None input returns empty dict."""
        from vectordb.haystack.metadata_filtering.common.config import (
            load_metadata_filtering_config,
        )

        result = load_metadata_filtering_config({})
        assert result == {}
