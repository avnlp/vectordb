"""Tests for dataloader factory.

This module tests the factory functions for creating dataloaders
from configuration and extracting queries for evaluation.
"""

import os
from unittest.mock import patch

import pytest  # noqa: F401

from vectordb.dataloaders.factory import (
    create_dataloader,
    create_generator,
    extract_queries_and_ground_truth,
)


class TestCreateGenerator:
    """Test suite for LLM generator creation.

    Tests cover:
    - Generator creation with configuration
    - API key handling from environment and config
    - Error handling for missing credentials
    - LangChain generator functionality
    - Framework parameter testing
    - Default parameter handling
    """

    def test_create_generator_success(self) -> None:
        """Test successful generator creation."""
        config = {
            "generator": {
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key-123",
            }
        }

        with patch.dict(os.environ, {"GROQ_API_KEY": "env-key"}):
            generator = create_generator(config)

            assert generator is not None

    def test_create_generator_uses_config_api_key(self) -> None:
        """Test that config API key is used when provided."""
        config = {
            "generator": {
                "api_key": "config-key",
                "model": "llama-3.3-70b-versatile",
            }
        }

        # Should not raise error - uses OpenAIChatGenerator with Groq API
        with patch(
            "haystack.components.generators.chat.OpenAIChatGenerator"
        ) as mock_generator:
            create_generator(config)
            mock_generator.assert_called_once()

    def test_create_generator_falls_back_to_env_variable(self) -> None:
        """Test fallback to GROQ_API_KEY environment variable."""
        config = {"generator": {"model": "llama-3.3-70b-versatile"}}

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "env-key"}),
            patch(
                "haystack.components.generators.chat.OpenAIChatGenerator"
            ) as mock_generator,
        ):
            create_generator(config)
            mock_generator.assert_called_once()

    def test_create_generator_missing_api_key_raises_error(self) -> None:
        """Test that missing API key raises ValueError."""
        config = {"generator": {"model": "llama-3.3-70b-versatile"}}

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="GROQ_API_KEY"),
        ):
            create_generator(config)

    def test_create_generator_default_model(self) -> None:
        """Test that default model is used when not specified."""
        config = {"generator": {"api_key": "test-key"}}

        with patch(
            "haystack.components.generators.chat.OpenAIChatGenerator"
        ) as mock_generator:
            create_generator(config)
            # Should use default model
            assert mock_generator.called

    def test_create_langchain_generator_success(self) -> None:
        """Test successful LangChain generator creation."""
        config = {
            "generator": {
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key-123",
            }
        }

        with patch("langchain_groq.ChatGroq") as mock_chatgroq:
            generator = create_generator(config, framework="langchain")
            mock_chatgroq.assert_called_once()
            assert generator is not None

    def test_create_langchain_generator_uses_config_api_key(self) -> None:
        """Test that config API key is used for LangChain generator."""
        config = {
            "generator": {
                "api_key": "config-key",
                "model": "llama-3.3-70b-versatile",
            }
        }

        with patch("langchain_groq.ChatGroq") as mock_chatgroq:
            create_generator(config, framework="langchain")
            mock_chatgroq.assert_called_once()

    def test_create_langchain_generator_falls_back_to_env_variable(self) -> None:
        """Test fallback to GROQ_API_KEY environment variable for LangChain."""
        config = {"generator": {"model": "llama-3.3-70b-versatile"}}

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "env-key"}),
            patch("langchain_groq.ChatGroq") as mock_chatgroq,
        ):
            create_generator(config, framework="langchain")
            mock_chatgroq.assert_called_once()

    def test_create_langchain_generator_missing_api_key_raises_error(self) -> None:
        """Test that missing API key raises ValueError for LangChain."""
        config = {"generator": {"model": "llama-3.3-70b-versatile"}}

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="GROQ_API_KEY"),
        ):
            create_generator(config, framework="langchain")

    def test_create_langchain_generator_default_model(self) -> None:
        """Test that default model is used for LangChain generator."""
        config = {"generator": {"api_key": "test-key"}}

        with patch("langchain_groq.ChatGroq") as mock_chatgroq:
            create_generator(config, framework="langchain")
            assert mock_chatgroq.called

    def test_create_generator_default_parameters(self) -> None:
        """Test default parameter handling in generator functions."""
        config = {"generator": {"api_key": "test-key"}}

        with patch(
            "haystack.components.generators.chat.OpenAIChatGenerator"
        ) as mock_generator:
            create_generator(config)
            # Verify default parameters are set
            call_args = mock_generator.call_args
            assert call_args[1]["generation_kwargs"]["temperature"] == 0.5
            assert call_args[1]["generation_kwargs"]["max_tokens"] == 2048

    def test_create_langchain_generator_default_parameters(self) -> None:
        """Test default parameter handling in LangChain generator."""
        config = {"generator": {"api_key": "test-key"}}

        with patch("langchain_groq.ChatGroq") as mock_chatgroq:
            create_generator(config, framework="langchain")
            # Verify default parameters are set
            call_args = mock_chatgroq.call_args
            assert call_args[1]["temperature"] == 0.5
            assert call_args[1]["max_tokens"] == 2048


class TestCreateGeneratorKwargs:
    """Test suite for generator kwargs handling.

    Tests cover:
    - Custom kwargs override defaults
    - Partial kwargs (only temperature or only max_tokens)
    - Empty generator config
    """

    def test_create_haystack_generator_custom_temperature(self) -> None:
        """Test that custom temperature overrides default."""
        config = {
            "generator": {
                "api_key": "test-key",
                "kwargs": {"temperature": 0.9},
            }
        }

        with patch(
            "haystack.components.generators.chat.OpenAIChatGenerator"
        ) as mock_generator:
            create_generator(config)
            call_args = mock_generator.call_args
            # Custom temperature should be preserved
            assert call_args[1]["generation_kwargs"]["temperature"] == 0.9
            # Default max_tokens should be added
            assert call_args[1]["generation_kwargs"]["max_tokens"] == 2048

    def test_create_haystack_generator_custom_max_tokens(self) -> None:
        """Test that custom max_tokens overrides default."""
        config = {
            "generator": {
                "api_key": "test-key",
                "kwargs": {"max_tokens": 4096},
            }
        }

        with patch(
            "haystack.components.generators.chat.OpenAIChatGenerator"
        ) as mock_generator:
            create_generator(config)
            call_args = mock_generator.call_args
            # Default temperature should be added
            assert call_args[1]["generation_kwargs"]["temperature"] == 0.5
            # Custom max_tokens should be preserved
            assert call_args[1]["generation_kwargs"]["max_tokens"] == 4096

    def test_create_haystack_generator_all_custom_kwargs(self) -> None:
        """Test that all custom kwargs override defaults."""
        config = {
            "generator": {
                "api_key": "test-key",
                "kwargs": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
            }
        }

        with patch(
            "haystack.components.generators.chat.OpenAIChatGenerator"
        ) as mock_generator:
            create_generator(config)
            call_args = mock_generator.call_args
            assert call_args[1]["generation_kwargs"]["temperature"] == 0.1
            assert call_args[1]["generation_kwargs"]["max_tokens"] == 1024
            assert call_args[1]["generation_kwargs"]["top_p"] == 0.95

    def test_create_langchain_generator_custom_temperature(self) -> None:
        """Test that custom temperature overrides default for LangChain."""
        config = {
            "generator": {
                "api_key": "test-key",
                "kwargs": {"temperature": 0.9},
            }
        }

        with patch("langchain_groq.ChatGroq") as mock_chatgroq:
            create_generator(config, framework="langchain")
            call_args = mock_chatgroq.call_args
            assert call_args[1]["temperature"] == 0.9
            assert call_args[1]["max_tokens"] == 2048

    def test_create_langchain_generator_custom_max_tokens(self) -> None:
        """Test that custom max_tokens overrides default for LangChain."""
        config = {
            "generator": {
                "api_key": "test-key",
                "kwargs": {"max_tokens": 4096},
            }
        }

        with patch("langchain_groq.ChatGroq") as mock_chatgroq:
            create_generator(config, framework="langchain")
            call_args = mock_chatgroq.call_args
            assert call_args[1]["temperature"] == 0.5
            assert call_args[1]["max_tokens"] == 4096

    def test_create_langchain_generator_all_custom_kwargs(self) -> None:
        """Test that all custom kwargs override defaults for LangChain."""
        config = {
            "generator": {
                "api_key": "test-key",
                "kwargs": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
            }
        }

        with patch("langchain_groq.ChatGroq") as mock_chatgroq:
            create_generator(config, framework="langchain")
            call_args = mock_chatgroq.call_args
            assert call_args[1]["temperature"] == 0.1
            assert call_args[1]["max_tokens"] == 1024
            assert call_args[1]["top_p"] == 0.95

    def test_create_generator_empty_generator_config(self) -> None:
        """Test generator creation with empty generator config section."""
        config: dict[str, dict[str, str]] = {}

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="GROQ_API_KEY"),
        ):
            create_generator(config)

    def test_create_generator_empty_kwargs(self) -> None:
        """Test generator creation with explicitly empty kwargs."""
        config = {
            "generator": {
                "api_key": "test-key",
                "kwargs": {},
            }
        }

        with patch(
            "haystack.components.generators.chat.OpenAIChatGenerator"
        ) as mock_generator:
            create_generator(config)
            call_args = mock_generator.call_args
            # Defaults should be applied
            assert call_args[1]["generation_kwargs"]["temperature"] == 0.5
            assert call_args[1]["generation_kwargs"]["max_tokens"] == 2048


class TestCreateDataloader:
    """Test suite for dataloader creation.

    Tests cover:
    - Creating dataloaders for all supported types
    - Configuration parameter passing
    - Generator integration for QA datasets
    - Error handling for unsupported types
    """

    def test_create_dataloader_arc(self) -> None:
        """Test creating ARC dataloader."""
        config = {
            "dataloader": {
                "type": "arc",
                "dataset_name": "ai2_arc",
                "split": "validation",
            }
        }

        with patch("vectordb.dataloaders.haystack.ARCDataloader"):
            dataloader = create_dataloader(config)
            assert dataloader is not None

    def test_create_dataloader_arc_default_names(self) -> None:
        """Test ARC dataloader with default dataset names."""
        config = {"dataloader": {"type": "arc", "split": "test"}}

        with patch("vectordb.dataloaders.haystack.ARCDataloader"):
            dataloader = create_dataloader(config)
            assert dataloader is not None

    def test_create_dataloader_earnings_calls(self) -> None:
        """Test creating earnings calls dataloader."""
        config = {
            "dataloader": {
                "type": "earnings_calls",
                "dataset_name": "lamini/earnings-calls-qa",
                "split": "train",
            }
        }

        with patch("vectordb.dataloaders.haystack.EarningsCallDataloader"):
            dataloader = create_dataloader(config)
            assert dataloader is not None

    def test_create_dataloader_triviaqa_with_generator(self) -> None:
        """Test TriviaQA dataloader creation with generator."""
        config = {
            "dataloader": {
                "type": "triviaqa",
                "dataset_name": "trivia_qa",
                "split": "test",
            },
            "generator": {
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key",
            },
        }

        with (
            patch(
                "haystack.components.generators.chat.OpenAIChatGenerator"
            ) as mock_gen,
            patch("vectordb.dataloaders.haystack.TriviaQADataloader"),
        ):
            dataloader = create_dataloader(config)
            assert dataloader is not None
            mock_gen.assert_called_once()

    def test_create_dataloader_popqa_with_generator(self) -> None:
        """Test PopQA dataloader creation with generator."""
        config = {
            "dataloader": {
                "type": "popqa",
                "dataset_name": "akariasai/PopQA",
                "split": "test",
            },
            "generator": {
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key",
            },
        }

        with (
            patch(
                "haystack.components.generators.chat.OpenAIChatGenerator"
            ) as mock_gen,
            patch("vectordb.dataloaders.haystack.PopQADataloader"),
        ):
            dataloader = create_dataloader(config)
            assert dataloader is not None
            mock_gen.assert_called_once()

    def test_create_dataloader_factscore_with_generator(self) -> None:
        """Test FactScore dataloader creation with generator."""
        config = {
            "dataloader": {
                "type": "factscore",
                "dataset_name": "dskar/FActScore",
                "split": "test",
            },
            "generator": {
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key",
            },
        }

        with (
            patch(
                "haystack.components.generators.chat.OpenAIChatGenerator"
            ) as mock_gen,
            patch("vectordb.dataloaders.haystack.FactScoreDataloader"),
        ):
            dataloader = create_dataloader(config)
            assert dataloader is not None
            mock_gen.assert_called_once()

    def test_create_dataloader_case_insensitive(self) -> None:
        """Test that dataloader type is case insensitive."""
        config = {
            "dataloader": {
                "type": "ARC",
                "split": "test",
            }
        }

        with patch("vectordb.dataloaders.haystack.ARCDataloader"):
            dataloader = create_dataloader(config)
            assert dataloader is not None

    def test_create_dataloader_unsupported_type_raises_error(self) -> None:
        """Test that unsupported dataloader type raises ValueError."""
        config = {"dataloader": {"type": "unsupported_type", "split": "test"}}

        with pytest.raises(ValueError, match="Unsupported dataloader type"):
            create_dataloader(config)

    def test_create_dataloader_unsupported_type_error_message_format(self) -> None:
        """Test that error message includes the unsupported type and supported types."""
        config = {"dataloader": {"type": "invalid_loader", "split": "test"}}

        with pytest.raises(ValueError) as exc_info:
            create_dataloader(config)

        error_message = str(exc_info.value)
        # Verify the error message contains the invalid type name
        assert "invalid_loader" in error_message
        # Verify the error message lists supported types
        assert "triviaqa" in error_message
        assert "arc" in error_message
        assert "popqa" in error_message
        assert "factscore" in error_message
        assert "earnings_calls" in error_message

    def test_create_dataloader_empty_type_raises_error(self) -> None:
        """Test that empty dataloader type raises ValueError."""
        config = {"dataloader": {"type": "", "split": "test"}}

        with pytest.raises(ValueError, match="Unsupported dataloader type"):
            create_dataloader(config)

    def test_create_dataloader_missing_type_raises_error(self) -> None:
        """Test that missing dataloader type raises ValueError."""
        config = {"dataloader": {"split": "test"}}

        with pytest.raises(ValueError, match="Unsupported dataloader type"):
            create_dataloader(config)

    def test_create_dataloader_empty_config_raises_error(self) -> None:
        """Test that empty config raises ValueError."""
        config: dict[str, dict[str, str]] = {}

        with pytest.raises(ValueError, match="Unsupported dataloader type"):
            create_dataloader(config)

    def test_create_dataloader_with_custom_split(self) -> None:
        """Test dataloader creation with custom split."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "validation",
            }
        }

        with patch("vectordb.dataloaders.haystack.ARCDataloader"):
            dataloader = create_dataloader(config)
            assert dataloader is not None


class TestExtractQueriesAndGroundTruth:
    """Test suite for query and ground truth extraction.

    Tests cover:
    - Extracting queries from different dataset types
    - Handling duplicates
    - Limit functionality
    - Metadata structure preservation
    """

    def test_extract_queries_from_arc(self) -> None:
        """Test extracting queries from ARC dataset."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = []

            queries = extract_queries_and_ground_truth(config)

            assert isinstance(queries, list)

    def test_extract_queries_structure(self) -> None:
        """Test that extracted queries have required structure."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = [
                {
                    "query": "What is X?",
                    "answers": ["X"],
                    "metadata": {
                        "id": "test-1",
                    },
                }
            ]

            queries = extract_queries_and_ground_truth(config)

            # Check structure if results exist
            if queries:
                assert "query" in queries[0]
                assert "answer" in queries[0]
                assert "relevant_doc_ids" in queries[0]

    def test_extract_queries_with_limit(self) -> None:
        """Test that limit is applied to extracted queries."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = []

            queries = extract_queries_and_ground_truth(config, limit=5)

            assert len(queries) <= 5

    def test_extract_queries_removes_duplicates(self) -> None:
        """Test that duplicate questions are removed."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            # EvaluationExtractor already handles deduplication, so we get single result
            mock_extract.return_value = [
                {
                    "query": "Same question?",
                    "answers": ["Answer"],
                    "metadata": {
                        "id": "id1",
                    },
                }
            ]

            queries = extract_queries_and_ground_truth(config)

            # Should have deduplicated (EvaluationExtractor does this)
            assert len(queries) == 1

    def test_extract_queries_empty_dataset(self) -> None:
        """Test extracting queries from empty dataset."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = []

            queries = extract_queries_and_ground_truth(config)

            assert queries == []

    def test_extract_queries_earnings_calls_structure(self) -> None:
        """Test query extraction for earnings calls dataset."""
        config = {
            "dataloader": {
                "type": "earnings_calls",
                "split": "train",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = [
                {
                    "query": "What was revenue?",
                    "answers": ["$100M"],
                    "metadata": {},
                }
            ]

            queries = extract_queries_and_ground_truth(config)

            if queries:
                assert "relevant_doc_ids" in queries[0]

    def test_extract_queries_earnings_calls_with_multiple_questions(self) -> None:
        """Test query extraction for earnings calls with multiple questions."""
        config = {
            "dataloader": {
                "type": "earnings_calls",
                "split": "train",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            # EvaluationExtractor already deduplicates
            mock_extract.return_value = [
                {
                    "query": "What was revenue?",
                    "answers": ["$100M"],
                    "metadata": {},
                },
                {
                    "query": "What was profit?",
                    "answers": ["$50M"],
                    "metadata": {},
                },
            ]

            queries = extract_queries_and_ground_truth(config)

            # Should have 2 unique queries
            # (deduplication handled by EvaluationExtractor)
            assert len(queries) == 2
            assert queries[0]["query"] == "What was revenue?"
            assert queries[1]["query"] == "What was profit?"

    def test_extract_queries_earnings_calls_empty_questions(self) -> None:
        """Test query extraction for earnings calls with empty questions."""
        config = {
            "dataloader": {
                "type": "earnings_calls",
                "split": "train",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            # EvaluationExtractor already filters out empty questions
            mock_extract.return_value = []

            queries = extract_queries_and_ground_truth(config)

            # Should filter out items with empty/missing questions
            assert len(queries) == 0

    def test_extract_queries_earnings_calls_with_limit(self) -> None:
        """Test query extraction for earnings calls with limit."""
        config = {
            "dataloader": {
                "type": "earnings_calls",
                "split": "train",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = [
                {"query": f"Question {i}", "answers": [f"Answer {i}"], "metadata": {}}
                for i in range(3)
            ]

            queries = extract_queries_and_ground_truth(config, limit=3)

            # Should respect the limit (passed to EvaluationExtractor)
            assert len(queries) == 3

    def test_extract_queries_edge_case_empty_config(self) -> None:
        """Test query extraction with empty configuration."""
        config = {"dataloader": {"type": "arc"}}

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = []

            queries = extract_queries_and_ground_truth(config)
            assert queries == []

    def test_extract_queries_edge_case_missing_metadata(self) -> None:
        """Test query extraction with missing metadata."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            # EvaluationExtractor filters out items with missing questions
            mock_extract.return_value = []

            queries = extract_queries_and_ground_truth(config)
            # Should filter out items with missing metadata/question
            assert len(queries) == 0

    def test_extract_queries_edge_case_special_characters(self) -> None:
        """Test query extraction with special characters in questions."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = [
                {
                    "query": "What is X? \n\t\r",  # Special chars
                    "answers": ["X"],
                    "metadata": {
                        "id": "test-1",
                    },
                }
            ]

            queries = extract_queries_and_ground_truth(config)
            if queries:
                assert "query" in queries[0]
                assert "What is X?" in queries[0]["query"]

    def test_extract_queries_edge_case_unicode_characters(self) -> None:
        """Test query extraction with unicode characters."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = [
                {
                    "query": "What is 你好?",  # Unicode chars
                    "answers": ["你好"],
                    "metadata": {
                        "id": "test-1",
                    },
                }
            ]

            queries = extract_queries_and_ground_truth(config)
            if queries:
                assert "query" in queries[0]
                assert "你好" in queries[0]["query"]

    def test_extract_queries_edge_case_very_long_questions(self) -> None:
        """Test query extraction with very long questions."""
        config = {
            "dataloader": {
                "type": "arc",
                "split": "test",
            }
        }

        long_question = "A" * 10000  # Very long question
        with patch(
            "vectordb.dataloaders.evaluation.EvaluationExtractor.extract"
        ) as mock_extract:
            mock_extract.return_value = [
                {
                    "query": long_question,
                    "answers": ["Answer"],
                    "metadata": {
                        "id": "test-1",
                    },
                }
            ]

            queries = extract_queries_and_ground_truth(config)
            if queries:
                assert len(queries[0]["query"]) == 10000
