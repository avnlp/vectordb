"""Tests for query enhancement LLM utilities."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.query_enhancement.utils.llm import (
    HYDE_PROMPT,
    MULTI_QUERY_PROMPT,
    STEP_BACK_PROMPT,
    create_groq_generator,
)


class TestCreateGroqGenerator:
    """Tests for create_groq_generator function."""

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    def test_with_api_key_in_config(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test with GROQ_API_KEY in config (should use config value)."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config = {
            "query_enhancement": {
                "llm": {
                    "api_key": "config-api-key-123",
                }
            }
        }
        result = create_groq_generator(config)

        mock_secret.from_token.assert_called_once_with("config-api-key-123")
        mock_generator_class.assert_called_once_with(
            api_key=mock_token,
            model="llama-3.3-70b-versatile",
            api_base_url="https://api.groq.com/openai/v1",
            generation_kwargs={"temperature": 0.7, "max_tokens": 1024},
        )
        assert result == mock_instance

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    @patch.dict(os.environ, {"GROQ_API_KEY": "env-api-key-456"})
    def test_with_api_key_as_env_var(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test with GROQ_API_KEY as env var (fallback)."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config: dict[str, Any] = {}
        result = create_groq_generator(config)

        mock_secret.from_token.assert_called_once_with("env-api-key-456")
        assert result == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_value_error(self) -> None:
        """Test missing GROQ_API_KEY raises ValueError."""
        # Ensure GROQ_API_KEY is not in environment
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]

        config: dict[str, Any] = {}

        with pytest.raises(ValueError, match="GROQ_API_KEY required"):
            create_groq_generator(config)

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    def test_custom_model_in_config(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test custom model in config."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config = {
            "query_enhancement": {
                "llm": {
                    "api_key": "test-key",
                    "model": "mixtral-8x7b-32768",
                }
            }
        }
        create_groq_generator(config)

        mock_generator_class.assert_called_once()
        call_kwargs = mock_generator_class.call_args[1]
        assert call_kwargs["model"] == "mixtral-8x7b-32768"

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    def test_default_model_when_not_provided(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test default model when not provided."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config = {
            "query_enhancement": {
                "llm": {
                    "api_key": "test-key",
                }
            }
        }
        create_groq_generator(config)

        call_kwargs = mock_generator_class.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    def test_generation_kwargs_defaults(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test generation_kwargs defaults (temperature=0.7, max_tokens=1024)."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config = {
            "query_enhancement": {
                "llm": {
                    "api_key": "test-key",
                }
            }
        }
        create_groq_generator(config)

        call_kwargs = mock_generator_class.call_args[1]
        assert call_kwargs["generation_kwargs"]["temperature"] == 0.7
        assert call_kwargs["generation_kwargs"]["max_tokens"] == 1024

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    def test_custom_generation_kwargs(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test custom generation_kwargs."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config = {
            "query_enhancement": {
                "llm": {
                    "api_key": "test-key",
                    "kwargs": {
                        "temperature": 0.5,
                        "max_tokens": 2048,
                        "top_p": 0.9,
                    },
                }
            }
        }
        create_groq_generator(config)

        call_kwargs = mock_generator_class.call_args[1]
        assert call_kwargs["generation_kwargs"]["temperature"] == 0.5
        assert call_kwargs["generation_kwargs"]["max_tokens"] == 2048
        assert call_kwargs["generation_kwargs"]["top_p"] == 0.9

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    def test_custom_generation_kwargs_partial_override(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test custom generation_kwargs with partial override (only temperature)."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config = {
            "query_enhancement": {
                "llm": {
                    "api_key": "test-key",
                    "kwargs": {
                        "temperature": 0.3,
                    },
                }
            }
        }
        create_groq_generator(config)

        call_kwargs = mock_generator_class.call_args[1]
        # Custom temperature
        assert call_kwargs["generation_kwargs"]["temperature"] == 0.3
        # Default max_tokens should be applied
        assert call_kwargs["generation_kwargs"]["max_tokens"] == 1024

    @patch("vectordb.haystack.query_enhancement.utils.llm.OpenAIChatGenerator")
    @patch("vectordb.haystack.query_enhancement.utils.llm.Secret")
    def test_api_base_url_is_set_correctly(
        self, mock_secret: MagicMock, mock_generator_class: MagicMock
    ) -> None:
        """Test api_base_url is set correctly."""
        mock_token = MagicMock()
        mock_secret.from_token.return_value = mock_token
        mock_instance = MagicMock()
        mock_generator_class.return_value = mock_instance

        config = {
            "query_enhancement": {
                "llm": {
                    "api_key": "test-key",
                }
            }
        }
        create_groq_generator(config)

        call_kwargs = mock_generator_class.call_args[1]
        assert call_kwargs["api_base_url"] == "https://api.groq.com/openai/v1"


class TestPromptTemplates:
    """Tests for prompt template constants."""

    def test_multi_query_prompt_exists(self) -> None:
        """Test MULTI_QUERY_PROMPT exists."""
        assert MULTI_QUERY_PROMPT is not None
        assert isinstance(MULTI_QUERY_PROMPT, str)

    def test_multi_query_prompt_contains_placeholders(self) -> None:
        """Test MULTI_QUERY_PROMPT contains {num_queries} and {query}."""
        assert "{num_queries}" in MULTI_QUERY_PROMPT
        assert "{query}" in MULTI_QUERY_PROMPT

    def test_hyde_prompt_exists(self) -> None:
        """Test HYDE_PROMPT exists."""
        assert HYDE_PROMPT is not None
        assert isinstance(HYDE_PROMPT, str)

    def test_hyde_prompt_contains_placeholders(self) -> None:
        """Test HYDE_PROMPT contains {num_docs} and {query}."""
        assert "{num_docs}" in HYDE_PROMPT
        assert "{query}" in HYDE_PROMPT

    def test_step_back_prompt_exists(self) -> None:
        """Test STEP_BACK_PROMPT exists."""
        assert STEP_BACK_PROMPT is not None
        assert isinstance(STEP_BACK_PROMPT, str)

    def test_step_back_prompt_contains_placeholders(self) -> None:
        """Test STEP_BACK_PROMPT contains {query}."""
        assert "{query}" in STEP_BACK_PROMPT

    def test_multi_query_prompt_can_be_formatted(self) -> None:
        """Test MULTI_QUERY_PROMPT can be formatted with expected args."""
        formatted = MULTI_QUERY_PROMPT.format(num_queries=3, query="test question")
        assert "3" in formatted
        assert "test question" in formatted

    def test_hyde_prompt_can_be_formatted(self) -> None:
        """Test HYDE_PROMPT can be formatted with expected args."""
        formatted = HYDE_PROMPT.format(num_docs=2, query="another test")
        assert "2" in formatted
        assert "another test" in formatted

    def test_step_back_prompt_can_be_formatted(self) -> None:
        """Test STEP_BACK_PROMPT can be formatted with expected args."""
        formatted = STEP_BACK_PROMPT.format(query="specific question")
        assert "specific question" in formatted
