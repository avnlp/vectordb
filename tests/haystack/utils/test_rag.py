"""Tests for RAGHelper utility class."""

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.utils.rag import RAGHelper


class TestRAGHelper:
    """Tests for RAGHelper class."""

    def test_create_generator_disabled(self) -> None:
        """Test generator returns None when RAG disabled."""
        config = {"rag": {"enabled": False}}
        result = RAGHelper.create_generator(config)
        assert result is None

    def test_create_generator_no_rag_section(self) -> None:
        """Test generator returns None when no RAG section."""
        config = {}
        result = RAGHelper.create_generator(config)
        assert result is None

    def test_create_generator_no_api_key(self) -> None:
        """Test error when RAG enabled but no API key."""
        os.environ.pop("GROQ_API_KEY", None)
        config = {"rag": {"enabled": True, "model": "test-model"}}
        with pytest.raises(ValueError, match="no API key provided"):
            RAGHelper.create_generator(config)

    @patch("vectordb.haystack.utils.rag.OpenAIGenerator")
    def test_create_generator_with_config_key(
        self, mock_generator_class: MagicMock
    ) -> None:
        """Test generator creation with API key in config."""
        config = {
            "rag": {
                "enabled": True,
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key",
            }
        }
        RAGHelper.create_generator(config)

        mock_generator_class.assert_called_once()
        call_kwargs = mock_generator_class.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    @patch.dict(os.environ, {"GROQ_API_KEY": "env-key"})
    @patch("vectordb.haystack.utils.rag.OpenAIGenerator")
    def test_create_generator_with_env_key(
        self, mock_generator_class: MagicMock
    ) -> None:
        """Test generator uses env var when no config key."""
        config = {"rag": {"enabled": True, "model": "test-model"}}
        RAGHelper.create_generator(config)

        call_kwargs = mock_generator_class.call_args[1]
        assert call_kwargs["api_key"] == "env-key"

    def test_format_prompt_default_template(self) -> None:
        """Test prompt formatting with default template."""
        docs = [
            Document(content="First document content."),
            Document(content="Second document content."),
        ]
        prompt = RAGHelper.format_prompt("What is the answer?", docs)

        assert "What is the answer?" in prompt
        assert "First document content." in prompt
        assert "Second document content." in prompt
        assert "Document 1:" in prompt
        assert "Document 2:" in prompt

    def test_format_prompt_custom_template(self) -> None:
        """Test prompt formatting with custom template."""
        docs = [Document(content="Some content")]
        template = "Context: {context}\nQ: {query}"
        prompt = RAGHelper.format_prompt("My question", docs, template)

        assert prompt == "Context: Document 1:\nSome content\nQ: My question"

    def test_format_prompt_empty_documents(self) -> None:
        """Test prompt formatting with no documents."""
        prompt = RAGHelper.format_prompt("Question?", [])
        assert "Question?" in prompt

    def test_generate(self) -> None:
        """Test RAG generation."""
        mock_generator = MagicMock()
        mock_generator.run.return_value = {"replies": ["Generated answer"]}

        docs = [Document(content="Context")]
        result = RAGHelper.generate(mock_generator, "Question?", docs)

        assert result == "Generated answer"
        mock_generator.run.assert_called_once()

    def test_generate_empty_replies(self) -> None:
        """Test generation with empty replies."""
        mock_generator = MagicMock()
        mock_generator.run.return_value = {"replies": []}

        result = RAGHelper.generate(mock_generator, "Question?", [])
        assert result == ""
