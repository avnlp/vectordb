"""Tests for query enhancement configuration loading and validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from vectordb.haystack.query_enhancement.utils.config import (
    load_config,
    resolve_env_vars,
    validate_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_with_dict_passthrough(self) -> None:
        """Dict input should be passed through unchanged (after env var resolution)."""
        config = {"key": "value", "nested": {"inner": "data"}}
        result = load_config(config)
        assert result == config

    def test_load_config_with_path_object(self, tmp_path: Path) -> None:
        """Path object to YAML file should load and parse correctly."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "dataloader": {"type": "triviaqa"},
            "embeddings": {"model": "test"},
        }
        config_file.write_text(yaml.dump(config_data))

        result = load_config(config_file)
        assert result == config_data

    def test_load_config_with_str_path(self, tmp_path: Path) -> None:
        """String path to YAML file should load and parse correctly."""
        config_file = tmp_path / "config.yaml"
        config_data = {"query_enhancement": {"method": "hyde"}}
        config_file.write_text(yaml.dump(config_data))

        result = load_config(str(config_file))
        assert result == config_data

    def test_load_config_nonexistent_file_raises_error(self, tmp_path: Path) -> None:
        """Non-existent file should raise FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(nonexistent)

    def test_load_config_nonexistent_str_path_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Non-existent str path should raise FileNotFoundError."""
        nonexistent = str(tmp_path / "missing.yaml")
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(nonexistent)

    def test_load_config_resolves_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables in config should be resolved."""
        monkeypatch.setenv("TEST_API_KEY", "secret123")
        config_file = tmp_path / "config.yaml"
        config_data = {"api_key": "${TEST_API_KEY}", "other": "plain"}
        config_file.write_text(yaml.dump(config_data))

        result = load_config(config_file)
        assert result["api_key"] == "secret123"
        assert result["other"] == "plain"

    def test_load_config_dict_resolves_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dict input should also have env vars resolved."""
        monkeypatch.setenv("MY_VAR", "resolved_value")
        config = {"setting": "${MY_VAR}"}

        result = load_config(config)
        assert result["setting"] == "resolved_value"


class TestResolveEnvVars:
    """Tests for resolve_env_vars function."""

    def test_resolve_env_vars_dict_with_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dict containing env var patterns should be resolved."""
        monkeypatch.setenv("DB_HOST", "localhost")
        config: dict[str, Any] = {"host": "${DB_HOST}", "port": 5432}

        result = resolve_env_vars(config)
        assert result["host"] == "localhost"
        assert result["port"] == 5432

    def test_resolve_env_vars_list_with_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """List containing env var patterns should be resolved."""
        monkeypatch.setenv("VAR1", "first")
        monkeypatch.setenv("VAR2", "second")
        config = ["${VAR1}", "${VAR2}", "static"]

        result = resolve_env_vars(config)
        assert result == ["first", "second", "static"]

    def test_resolve_env_vars_nested_structures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nested structures (dict in list, list in dict) should be resolved."""
        monkeypatch.setenv("NESTED_VAR", "nested_value")
        config: dict[str, Any] = {
            "outer": {
                "inner": "${NESTED_VAR}",
                "list_in_dict": ["${NESTED_VAR}", "static"],
            },
            "dict_in_list": [{"key": "${NESTED_VAR}"}],
        }

        result = resolve_env_vars(config)
        assert result["outer"]["inner"] == "nested_value"
        assert result["outer"]["list_in_dict"] == ["nested_value", "static"]
        assert result["dict_in_list"][0]["key"] == "nested_value"

    def test_resolve_env_vars_non_matching_strings(self) -> None:
        """Strings not matching ${...} pattern should be unchanged."""
        config = {
            "plain": "no_env_var",
            "partial_start": "${incomplete",
            "partial_end": "incomplete}",
            "dollar_only": "$VAR",
            "curly_only": "{VAR}",
        }

        result = resolve_env_vars(config)
        assert result["plain"] == "no_env_var"
        assert result["partial_start"] == "${incomplete"
        assert result["partial_end"] == "incomplete}"
        assert result["dollar_only"] == "$VAR"
        assert result["curly_only"] == "{VAR}"

    def test_resolve_env_vars_existing_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing env vars should resolve to their values."""
        monkeypatch.setenv("EXISTING_VAR", "my_value")
        result = resolve_env_vars("${EXISTING_VAR}")
        assert result == "my_value"

    def test_resolve_env_vars_missing_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing env vars should resolve to empty string."""
        # Ensure the var doesn't exist
        monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
        result = resolve_env_vars("${DEFINITELY_NOT_SET}")
        assert result == ""

    def test_resolve_env_vars_none_value(self) -> None:
        """None values should pass through unchanged."""
        result = resolve_env_vars(None)
        assert result is None

    def test_resolve_env_vars_integer_passthrough(self) -> None:
        """Integer values should pass through unchanged."""
        result = resolve_env_vars(42)
        assert result == 42

    def test_resolve_env_vars_float_passthrough(self) -> None:
        """Float values should pass through unchanged."""
        result = resolve_env_vars(3.14)
        assert result == 3.14

    def test_resolve_env_vars_bool_passthrough(self) -> None:
        """Boolean values should pass through unchanged."""
        assert resolve_env_vars(True) is True
        assert resolve_env_vars(False) is False

    def test_resolve_env_vars_empty_dict(self) -> None:
        """Empty dict should return empty dict."""
        result = resolve_env_vars({})
        assert result == {}

    def test_resolve_env_vars_empty_list(self) -> None:
        """Empty list should return empty list."""
        result = resolve_env_vars([])
        assert result == []


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_config_all_required_sections(self) -> None:
        """Valid config with all required sections should not raise."""
        config = {
            "dataloader": {"type": "triviaqa"},
            "embeddings": {"model": "openai"},
            "query_enhancement": {"method": "hyde"},
        }
        # Should not raise
        validate_config(config)

    def test_validate_config_missing_dataloader(self) -> None:
        """Missing dataloader section should raise ValueError."""
        config = {
            "embeddings": {"model": "openai"},
            "query_enhancement": {"method": "hyde"},
        }
        with pytest.raises(
            ValueError, match="Missing required config section: dataloader"
        ):
            validate_config(config)

    def test_validate_config_missing_embeddings(self) -> None:
        """Missing embeddings section should raise ValueError."""
        config = {
            "dataloader": {"type": "triviaqa"},
            "query_enhancement": {"method": "hyde"},
        }
        with pytest.raises(
            ValueError, match="Missing required config section: embeddings"
        ):
            validate_config(config)

    def test_validate_config_missing_query_enhancement(self) -> None:
        """Missing query_enhancement section should raise ValueError."""
        config = {
            "dataloader": {"type": "triviaqa"},
            "embeddings": {"model": "openai"},
        }
        with pytest.raises(
            ValueError, match="Missing required config section: query_enhancement"
        ):
            validate_config(config)

    def test_validate_config_missing_multiple_sections(self) -> None:
        """Missing multiple sections should raise for the first missing one."""
        config = {"dataloader": {"type": "triviaqa"}}
        # Should raise for embeddings (first missing in order)
        with pytest.raises(
            ValueError, match="Missing required config section: embeddings"
        ):
            validate_config(config)

    def test_validate_config_empty_config(self) -> None:
        """Empty config should raise ValueError for first missing section."""
        with pytest.raises(
            ValueError, match="Missing required config section: dataloader"
        ):
            validate_config({})

    def test_validate_config_extra_sections_ok(self) -> None:
        """Extra sections beyond required ones should be allowed."""
        config = {
            "dataloader": {"type": "triviaqa"},
            "embeddings": {"model": "openai"},
            "query_enhancement": {"method": "hyde"},
            "extra_section": {"foo": "bar"},
            "another_extra": [1, 2, 3],
        }
        # Should not raise
        validate_config(config)

    def test_validate_config_empty_required_sections(self) -> None:
        """Empty required sections (but present) should not raise."""
        config = {
            "dataloader": {},
            "embeddings": {},
            "query_enhancement": {},
        }
        # Should not raise - we only check presence, not contents
        validate_config(config)

    def test_validate_config_none_values_in_sections(self) -> None:
        """None values in required sections should not raise (section exists)."""
        config = {
            "dataloader": None,
            "embeddings": None,
            "query_enhancement": None,
        }
        # Should not raise - keys exist
        validate_config(config)
