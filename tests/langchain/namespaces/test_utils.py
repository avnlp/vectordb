"""Tests for LangChain namespace config and data utilities."""

from unittest.mock import Mock, patch

import pytest
import yaml
from langchain_core.documents import Document

from vectordb.langchain.namespaces.utils.config import load_config, resolve_env_vars
from vectordb.langchain.namespaces.utils.data import (
    get_namespace_configs,
    load_documents_from_config,
)


class TestConfigUtils:
    """Tests for YAML config loading and env var resolution."""

    def test_load_config_resolves_env_vars_recursively(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Load a YAML config and resolve nested ${ENV_VAR} values."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "embedding": {
                "model": "mini-model",
                "api_key": "${TEST_API_KEY}",
                "nested": {"region": "${TEST_REGION}", "other": 3},
            },
            "plain": "value",
        }
        config_path.write_text(yaml.dump(config_data))
        monkeypatch.setenv("TEST_API_KEY", "secret-key")
        monkeypatch.setenv("TEST_REGION", "us-east-1")

        loaded = load_config(str(config_path))

        assert loaded["embedding"]["api_key"] == "secret-key"
        assert loaded["embedding"]["nested"]["region"] == "us-east-1"
        assert loaded["embedding"]["nested"]["other"] == 3
        assert loaded["plain"] == "value"

    def test_load_config_missing_file_raises(self) -> None:
        """Raise FileNotFoundError when config path does not exist."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/tmp/does-not-exist-config.yaml")

    def test_resolve_env_vars_uses_empty_string_when_missing(self) -> None:
        """Replace unresolved ${ENV_VAR} placeholders with empty strings."""
        config = {"token": "${MISSING_TOKEN}"}

        resolved = resolve_env_vars(config)

        assert resolved == {"token": ""}


class TestDataUtils:
    """Tests for loading documents and namespace definitions from config."""

    @patch("vectordb.langchain.namespaces.utils.data.DataloaderCatalog.create")
    def test_load_documents_from_config_with_overrides(self, mock_create) -> None:
        """Honor split/limit overrides when creating the dataloader."""
        mock_loader = Mock()
        mock_loaded_dataset = Mock()
        expected_documents = [Document(page_content="doc", metadata={"id": "1"})]
        mock_loaded_dataset.to_langchain.return_value = expected_documents
        mock_loader.load.return_value = mock_loaded_dataset
        mock_create.return_value = mock_loader

        config = {
            "dataset": {
                "type": "TriviaQA",
                "dataset_name": "rc",
                "split": "train",
                "limit": 25,
            }
        }

        documents = load_documents_from_config(
            config,
            split_override="validation",
            limit_override=5,
        )

        assert documents == expected_documents
        mock_create.assert_called_once_with(
            "triviaqa",
            split="validation",
            limit=5,
            dataset_id="rc",
        )
        mock_loader.load.assert_called_once_with()
        mock_loaded_dataset.to_langchain.assert_called_once_with()

    @patch("vectordb.langchain.namespaces.utils.data.DataloaderCatalog.create")
    def test_load_documents_from_config_uses_dataloader_fallback(
        self, mock_create
    ) -> None:
        """Use dataloader config section when dataset section is absent."""
        mock_loader = Mock()
        mock_loaded_dataset = Mock()
        mock_loaded_dataset.to_langchain.return_value = []
        mock_loader.load.return_value = mock_loaded_dataset
        mock_create.return_value = mock_loader

        config = {
            "dataloader": {
                "type": "ARC",
                "dataset_name": "ARC-Challenge",
                "split": "test",
                "limit": 10,
            }
        }

        load_documents_from_config(config)

        mock_create.assert_called_once_with(
            "arc",
            split="test",
            limit=10,
            dataset_id="ARC-Challenge",
        )

    def test_load_documents_from_config_missing_dataset_type_raises(self) -> None:
        """Raise ValueError when dataset type is not configured."""
        with pytest.raises(ValueError, match="Dataset type must be specified"):
            load_documents_from_config({"dataset": {"split": "test"}})

    def test_get_namespace_configs_present(self) -> None:
        """Return namespace definitions when provided."""
        config = {
            "namespaces": {
                "definitions": [
                    {"name": "train_ns", "split": "train"},
                    {"name": "test_ns", "split": "test"},
                ]
            }
        }

        assert get_namespace_configs(config) == [
            {"name": "train_ns", "split": "train"},
            {"name": "test_ns", "split": "test"},
        ]

    def test_get_namespace_configs_absent(self) -> None:
        """Return an empty list when namespace definitions are missing."""
        assert get_namespace_configs({"pipeline": {"name": "x"}}) == []
