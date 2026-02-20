"""Tests for namespace data utilities.

This module tests the data loading utilities for namespace pipelines,
including document loading from configuration and namespace config extraction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.namespaces.utils.data import (
    get_namespace_configs,
    load_documents_from_config,
)


class TestLoadDocumentsFromConfig:
    """Tests for load_documents_from_config function."""

    def test_load_documents_with_valid_dataset_config(self) -> None:
        """Test loading documents with a valid dataset configuration."""
        config = {
            "dataset": {
                "type": "triviaqa",
                "dataset_name": "trivia_qa",
                "split": "test",
                "limit": 10,
            }
        }

        sample_documents = [
            Document(content="Document 1 content", meta={"source": "test1"}),
            Document(content="Document 2 content", meta={"source": "test2"}),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            result = load_documents_from_config(config)

            mock_create.assert_called_once_with(
                "triviaqa",
                split="test",
                limit=10,
                dataset_id="trivia_qa",
            )

            assert len(result) == 2
            assert isinstance(result[0], Document)
            assert result[0].content == "Document 1 content"
            assert result[0].meta == {"source": "test1"}
            assert result[1].content == "Document 2 content"
            assert result[1].meta == {"source": "test2"}

    def test_load_documents_with_dataloader_key(self) -> None:
        """Test loading documents using 'dataloader' key instead of 'dataset'."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "train",
            }
        }

        sample_documents = [
            Document(content="ARC question 1", meta={"id": 1}),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            result = load_documents_from_config(config)

            mock_create.assert_called_once_with(
                "arc",
                split="train",
                limit=None,
                dataset_id=None,
            )

            assert len(result) == 1
            assert result[0].content == "ARC question 1"

    def test_load_documents_missing_dataset_type(self) -> None:
        """Test that ValueError is raised when dataset type is missing."""
        config = {
            "dataset": {
                "dataset_name": "some_dataset",
            }
        }

        with pytest.raises(
            ValueError, match="Dataset type must be specified in config"
        ):
            load_documents_from_config(config)

    def test_load_documents_empty_dataset_type(self) -> None:
        """Test that ValueError is raised when dataset type is empty string."""
        config = {
            "dataset": {
                "type": "",
            }
        }

        with pytest.raises(
            ValueError, match="Dataset type must be specified in config"
        ):
            load_documents_from_config(config)

    def test_load_documents_with_split_override(self) -> None:
        """Test that split_override parameter overrides config value."""
        config = {
            "dataset": {
                "type": "popqa",
                "split": "test",
            }
        }

        sample_documents = [Document(content="PopQA content", meta={})]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            load_documents_from_config(config, split_override="validation")

            mock_create.assert_called_once_with(
                "popqa",
                split="validation",
                limit=None,
                dataset_id=None,
            )

    def test_load_documents_with_limit_override(self) -> None:
        """Test that limit_override parameter overrides config value."""
        config = {
            "dataset": {
                "type": "factscore",
                "limit": 100,
            }
        }

        sample_documents = [Document(content="FactScore content", meta={})]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            load_documents_from_config(config, limit_override=50)

            mock_create.assert_called_once_with(
                "factscore",
                split="test",
                limit=50,
                dataset_id=None,
            )

    def test_load_documents_with_both_overrides(self) -> None:
        """Test using both split_override and limit_override together."""
        config = {
            "dataset": {
                "type": "earnings_calls",
                "split": "train",
                "limit": 1000,
            }
        }

        sample_documents = [
            Document(content="Earnings call transcript", meta={"company": "ABC"})
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            load_documents_from_config(config, split_override="test", limit_override=25)

            mock_create.assert_called_once_with(
                "earnings_calls",
                split="test",
                limit=25,
                dataset_id=None,
            )

    def test_load_documents_default_split(self) -> None:
        """Test that default split is 'test' when not specified."""
        config = {
            "dataset": {
                "type": "triviaqa",
            }
        }

        sample_documents = [Document(content="Content", meta={})]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            load_documents_from_config(config)

            mock_create.assert_called_once_with(
                "triviaqa",
                split="test",
                limit=None,
                dataset_id=None,
            )

    def test_load_documents_case_insensitive_type(self) -> None:
        """Test that dataset type is converted to lowercase."""
        config = {
            "dataset": {
                "type": "TriviaQA",
            }
        }

        sample_documents = [Document(content="Content", meta={})]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            load_documents_from_config(config)

            mock_create.assert_called_once_with(
                "triviaqa",
                split="test",
                limit=None,
                dataset_id=None,
            )

    def test_load_documents_without_metadata(self) -> None:
        """Test loading documents when metadata is not provided in data."""
        config = {
            "dataset": {
                "type": "arc",
            }
        }

        sample_documents = [
            Document(content="Document without metadata", meta={}),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            result = load_documents_from_config(config)

            assert len(result) == 1
            assert result[0].content == "Document without metadata"
            assert result[0].meta == {}

    def test_load_documents_empty_result(self) -> None:
        """Test handling empty dataset result."""
        config = {
            "dataset": {
                "type": "popqa",
            }
        }

        sample_documents = []

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.return_value = mock_loader

            result = load_documents_from_config(config)

            assert result == []

    def test_load_documents_registry_raises_error(self) -> None:
        """Test handling errors from DataloaderCatalog.create."""
        config = {
            "dataset": {
                "type": "unsupported_dataset",
            }
        }

        with patch(
            "vectordb.haystack.namespaces.utils.data.DataloaderCatalog.create"
        ) as mock_create:
            mock_create.side_effect = ValueError(
                "Unknown dataset: 'unsupported_dataset'"
            )

            with pytest.raises(ValueError, match="Unknown dataset"):
                load_documents_from_config(config)


class TestGetNamespaceConfigs:
    """Tests for get_namespace_configs function."""

    def test_get_namespace_configs_with_valid_config(self) -> None:
        """Test extracting namespace definitions from valid config."""
        config = {
            "namespaces": {
                "definitions": [
                    {"name": "namespace1", "description": "First namespace"},
                    {"name": "namespace2", "description": "Second namespace"},
                ]
            }
        }

        result = get_namespace_configs(config)

        assert len(result) == 2
        assert result[0]["name"] == "namespace1"
        assert result[1]["name"] == "namespace2"

    def test_get_namespace_configs_empty_definitions(self) -> None:
        """Test with empty namespace definitions list."""
        config = {"namespaces": {"definitions": []}}

        result = get_namespace_configs(config)

        assert result == []

    def test_get_namespace_configs_missing_namespaces_key(self) -> None:
        """Test when 'namespaces' key is missing from config."""
        config = {
            "pipeline": {
                "name": "test-pipeline",
            }
        }

        result = get_namespace_configs(config)

        assert result == []

    def test_get_namespace_configs_missing_definitions_key(self) -> None:
        """Test when 'definitions' key is missing from namespaces."""
        config = {
            "namespaces": {
                "default": "namespace1",
            }
        }

        result = get_namespace_configs(config)

        assert result == []

    def test_get_namespace_configs_empty_config(self) -> None:
        """Test with completely empty config."""
        config: dict[str, Any] = {}

        result = get_namespace_configs(config)

        assert result == []

    def test_get_namespace_configs_complex_definitions(self) -> None:
        """Test with complex namespace definitions containing nested data."""
        config = {
            "namespaces": {
                "definitions": [
                    {
                        "name": "financial",
                        "description": "Financial documents",
                        "embedding_model": "finance-model",
                        "metadata_filters": {"category": "finance"},
                    },
                    {
                        "name": "legal",
                        "description": "Legal documents",
                        "embedding_model": "legal-model",
                        "metadata_filters": {"category": "legal"},
                    },
                ]
            }
        }

        result = get_namespace_configs(config)

        assert len(result) == 2
        assert result[0]["name"] == "financial"
        assert result[0]["embedding_model"] == "finance-model"
        assert result[0]["metadata_filters"] == {"category": "finance"}
        assert result[1]["name"] == "legal"
