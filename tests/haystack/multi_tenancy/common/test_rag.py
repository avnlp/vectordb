"""Tests for multi-tenancy RAG utilities."""

import os
from unittest.mock import MagicMock, patch

from vectordb.haystack.multi_tenancy.common.rag import (
    create_rag_generator,
    create_rag_pipeline,
)


class TestCreateRAGGenerator:
    """Tests for create_rag_generator function."""

    def test_default_generator(self):
        """Test creating generator with default config."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {
                "generator": {
                    "api_key": "test_api_key",
                }
            }
            result = create_rag_generator(config)

            mock_class.assert_called_once_with(
                api_key="test_api_key",
                model="llama-3.3-70b-versatile",
                api_base_url="https://api.groq.com/openai/v1",
                generation_kwargs={
                    "temperature": 0.5,
                    "max_tokens": 2048,
                },
            )
            assert result == mock_instance

    def test_custom_model(self):
        """Test creating generator with custom model."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {
                "generator": {
                    "api_key": "test_api_key",
                    "model": "custom-model",
                }
            }
            create_rag_generator(config)

            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["model"] == "custom-model"

    def test_custom_api_base_url(self):
        """Test creating generator with custom API base URL."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {
                "generator": {
                    "api_key": "test_api_key",
                    "api_base_url": "https://custom.api.com/v1",
                }
            }
            create_rag_generator(config)

            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["api_base_url"] == "https://custom.api.com/v1"

    def test_api_key_from_environment(self):
        """Test that API key is read from environment variable."""
        os.environ["GROQ_API_KEY"] = "env_api_key"
        try:
            with patch(
                "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
            ) as mock_class:
                mock_instance = MagicMock()
                mock_class.return_value = mock_instance

                config = {"generator": {}}
                create_rag_generator(config)

                mock_class.assert_called_once()
                call_kwargs = mock_class.call_args[1]
                assert call_kwargs["api_key"] == "env_api_key"
        finally:
            del os.environ["GROQ_API_KEY"]

    def test_custom_generation_kwargs(self):
        """Test creating generator with custom generation kwargs."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {
                "generator": {
                    "api_key": "test_api_key",
                    "kwargs": {
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "top_p": 0.9,
                    },
                }
            }
            create_rag_generator(config)

            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["generation_kwargs"]["temperature"] == 0.7
            assert call_kwargs["generation_kwargs"]["max_tokens"] == 1000
            assert call_kwargs["generation_kwargs"]["top_p"] == 0.9

    def test_default_kwargs_when_not_specified(self):
        """Test that default kwargs are set when not specified."""
        with patch(
            "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            config = {
                "generator": {
                    "api_key": "test_api_key",
                }
            }
            create_rag_generator(config)

            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["generation_kwargs"]["temperature"] == 0.5
            assert call_kwargs["generation_kwargs"]["max_tokens"] == 2048


class TestCreateRAGPipeline:
    """Tests for create_rag_pipeline function."""

    def test_default_pipeline(self):
        """Test creating RAG pipeline with default config."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
            ) as mock_generator,
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.Pipeline"
            ) as mock_pipeline_class,
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.PromptBuilder"
            ) as mock_prompt_builder,
        ):
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_prompt_builder_instance = MagicMock()
            mock_prompt_builder.return_value = mock_prompt_builder_instance
            mock_generator_instance = MagicMock()
            mock_generator.return_value = mock_generator_instance

            config = {
                "generator": {
                    "api_key": "test_api_key",
                }
            }
            create_rag_pipeline(config)

            # Verify pipeline was created
            mock_pipeline_class.assert_called_once()

            # Verify components were added
            mock_pipeline.add_component.assert_any_call(
                "prompt_builder", mock_prompt_builder_instance
            )
            mock_pipeline.add_component.assert_any_call(
                "generator", mock_generator_instance
            )

            # Verify connection was made
            mock_pipeline.connect.assert_called_once_with(
                "prompt_builder.prompt", "generator.prompt"
            )

    def test_custom_prompt_template(self):
        """Test creating RAG pipeline with custom prompt template."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
            ) as mock_generator,
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.Pipeline"
            ) as mock_pipeline_class,
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.PromptBuilder"
            ) as mock_prompt_builder,
        ):
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_generator_instance = MagicMock()
            mock_generator.return_value = mock_generator_instance

            custom_template = "Custom: {{ query }}"
            config = {
                "generator": {
                    "api_key": "test_api_key",
                },
                "rag": {
                    "prompt_template": custom_template,
                },
            }
            create_rag_pipeline(config)

            # Verify custom template was used
            mock_prompt_builder.assert_called_once_with(template=custom_template)

    def test_pipeline_has_correct_components(self):
        """Test that pipeline has the correct components connected."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.OpenAIGenerator"
            ) as mock_generator,
            patch(
                "vectordb.haystack.multi_tenancy.common.rag.Pipeline"
            ) as mock_pipeline_class,
            patch("vectordb.haystack.multi_tenancy.common.rag.PromptBuilder"),
        ):
            mock_pipeline = MagicMock()
            mock_pipeline_class.return_value = mock_pipeline
            mock_generator_instance = MagicMock()
            mock_generator.return_value = mock_generator_instance

            config = {
                "generator": {
                    "api_key": "test_api_key",
                }
            }
            create_rag_pipeline(config)

            # Verify pipeline has 2 components
            assert mock_pipeline.add_component.call_count == 2
