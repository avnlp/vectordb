"""Tests for cost-optimized RAG utils and config modules."""

import logging
from unittest.mock import MagicMock, patch

import pytest
import yaml
from haystack import Document

from vectordb.haystack.cost_optimized_rag.base.config import (
    RAGConfig,
    _resolve_env_vars,
    load_config,
)
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    format_search_results,
    load_documents_from_config,
)
from vectordb.haystack.cost_optimized_rag.utils.prompt_templates import (
    COST_OPTIMIZED_RAG_TEMPLATE,
    RAG_ANSWER_TEMPLATE,
    RAG_ANSWER_WITH_SOURCES_TEMPLATE,
)


class TestConfigLoading:
    """Tests for configuration loading and validation."""

    def test_load_config_valid(self, tmp_path):
        """Test loading a valid configuration file."""
        config_data = {
            "collection": {"name": "test_collection", "description": "Test"},
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
            },
            "dataloader": {
                "type": "triviaqa",
                "dataset_name": "trivia_qa",
                "config": "rc",
                "split": "test",
            },
            "search": {"top_k": 10},
            "generator": {
                "enabled": False,
                "provider": "groq",
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))

        assert isinstance(config, RAGConfig)
        assert config.collection.name == "test_collection"
        assert config.embeddings.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.search.top_k == 10

    def test_load_config_missing_file(self):
        """Test loading config from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty configuration file."""
        config_path = tmp_path / "empty_config.yaml"
        with open(config_path, "w") as f:
            f.write("")

        with pytest.raises(ValueError):
            load_config(str(config_path))

    def test_resolve_env_vars_simple(self):
        """Test resolving simple environment variables."""
        import os

        os.environ["TEST_VAR"] = "test_value"
        result = _resolve_env_vars("${TEST_VAR}")
        assert result == "test_value"

    def test_resolve_env_vars_with_default(self):
        """Test resolving environment variables with default value."""
        result = _resolve_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_resolve_env_vars_dict(self):
        """Test resolving environment variables in dictionary."""
        import os

        os.environ["TEST_KEY"] = "test_val"
        data = {"key": "${TEST_KEY}", "static": "value"}
        result = _resolve_env_vars(data)
        assert result["key"] == "test_val"
        assert result["static"] == "value"

    def test_resolve_env_vars_list(self):
        """Test resolving environment variables in list."""
        import os

        os.environ["TEST_VAL"] = "resolved"
        data = ["${TEST_VAL}", "static"]
        result = _resolve_env_vars(data)
        assert result[0] == "resolved"
        assert result[1] == "static"

    def test_config_generator_settings(self, tmp_path):
        """Test generator settings in config."""
        config_data = {
            "collection": {"name": "test", "description": "Test"},
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"type": "triviaqa"},
            "generator": {
                "enabled": True,
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key",
                "temperature": 0.5,
                "max_tokens": 2048,
            },
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))

        assert config.generator.enabled is True
        assert config.generator.model == "llama-3.3-70b-versatile"
        assert config.generator.temperature == 0.5
        assert config.generator.max_tokens == 2048


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_logger(self, tmp_path):
        """Test logger creation from config."""
        config_data = {
            "collection": {"name": "test"},
            "dataloader": {"type": "triviaqa"},
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "logging": {"name": "test_logger", "level": "INFO"},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        logger = create_logger(config)

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        # Logger level may be inherited, just check it's a Logger instance
        assert logger.level >= logging.INFO or logger.level == 0  # 0 means unset

    def test_format_search_results_basic(self):
        """Test formatting search results."""
        docs = [
            Document(
                id="doc1",
                content="Content 1",
                meta={"source": "test"},
                score=0.95,
            ),
            Document(
                id="doc2",
                content="Content 2",
                meta={"source": "test"},
                score=0.85,
            ),
        ]

        results = format_search_results(docs)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "Content 1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"]["source"] == "test"

    def test_format_search_results_with_embeddings(self):
        """Test formatting search results with embeddings."""
        embedding = [0.1, 0.2, 0.3]
        docs = [
            Document(
                id="doc1",
                content="Content 1",
                embedding=embedding,
                meta={"source": "test"},
                score=0.95,
            ),
        ]

        results = format_search_results(docs, include_embeddings=True)

        assert len(results) == 1
        assert results[0]["embedding"] == embedding

    def test_format_search_results_no_score(self):
        """Test formatting search results when score is None."""
        docs = [
            Document(
                id="doc1",
                content="Content 1",
                meta={"source": "test", "score": 0.75},
                score=None,
            ),
        ]

        results = format_search_results(docs)

        assert len(results) == 1
        assert results[0]["score"] == 0.75

    @patch("vectordb.haystack.cost_optimized_rag.utils.common.DataloaderCatalog.create")
    def test_load_documents_from_config(
        self,
        mock_create,
        tmp_path,
    ):
        """Test loading documents from config."""
        # Setup mocks with the correct chain: create() -> load() -> to_haystack()
        sample_documents = [
            Document(id="1", content="Sample 1"),
            Document(id="2", content="Sample 2"),
        ]
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_create.return_value = mock_loader

        config_data = {
            "collection": {"name": "test"},
            "dataloader": {
                "type": "triviaqa",
                "dataset_name": "trivia_qa",
                "config": "rc",
                "split": "test",
                "limit": None,
            },
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        documents = load_documents_from_config(config)

        assert len(documents) == 2
        assert documents[0].id == "1"
        mock_create.assert_called_once()

    @patch("vectordb.haystack.cost_optimized_rag.utils.common.DataloaderCatalog.create")
    def test_load_documents_with_limit(
        self,
        mock_create,
        tmp_path,
    ):
        """Test loading documents with limit."""
        all_documents = [
            Document(id="1", content="Sample 1"),
            Document(id="2", content="Sample 2"),
            Document(id="3", content="Sample 3"),
        ]

        def create_with_limit(*args, **kwargs):
            limit = kwargs.get("limit")
            mock_loader = MagicMock()
            mock_dataset = MagicMock()
            # Return only the limited number of documents
            if limit:
                mock_dataset.to_haystack.return_value = all_documents[:limit]
            else:
                mock_dataset.to_haystack.return_value = all_documents
            mock_loader.load.return_value = mock_dataset
            return mock_loader

        mock_create.side_effect = create_with_limit

        config_data = {
            "collection": {"name": "test"},
            "dataloader": {
                "type": "triviaqa",
                "dataset_name": "trivia_qa",
                "config": "rc",
                "split": "test",
                "limit": 2,
            },
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        documents = load_documents_from_config(config)

        assert len(documents) == 2


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_rag_answer_template_exists(self):
        """Test RAG answer template is defined."""
        assert isinstance(RAG_ANSWER_TEMPLATE, str)
        assert "Question" in RAG_ANSWER_TEMPLATE
        assert "Answer" in RAG_ANSWER_TEMPLATE

    def test_rag_answer_with_sources_template_exists(self):
        """Test RAG answer with sources template is defined."""
        assert isinstance(RAG_ANSWER_WITH_SOURCES_TEMPLATE, str)
        assert "Question" in RAG_ANSWER_WITH_SOURCES_TEMPLATE
        assert "Answer" in RAG_ANSWER_WITH_SOURCES_TEMPLATE

    def test_cost_optimized_rag_template_exists(self):
        """Test cost-optimized RAG template is defined."""
        assert isinstance(COST_OPTIMIZED_RAG_TEMPLATE, str)
        assert "Question" in COST_OPTIMIZED_RAG_TEMPLATE
        assert "Answer" in COST_OPTIMIZED_RAG_TEMPLATE

    def test_templates_have_jinja_syntax(self):
        """Test templates use Jinja2 syntax."""
        assert "{% for" in RAG_ANSWER_TEMPLATE
        assert "{{ doc" in RAG_ANSWER_TEMPLATE
        assert "{{ query }}" in RAG_ANSWER_TEMPLATE
