"""Tests for Haystack query enhancement components.

This module tests the QueryEnhancer and QueryRouter components which handle
prompt engineering and query transformation for improved retrieval.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestQueryEnhancer:
    """Test suite for QueryEnhancer component.

    Tests cover:
    - Multi-query generation
    - Hypothetical document generation (HyDE)
    - Step-back query generation
    - Error handling and fallbacks
    """

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_initialization(self, mock_generator_class) -> None:
        """Test QueryEnhancer initialization."""
        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            # The generator property should be a mock instance
            assert hasattr(
                enhancer.generator, "run"
            )  # Check it has the expected methods
            # Check that the generator was initialized with the correct parameters
            mock_generator_class.assert_called_once()
            call_args = mock_generator_class.call_args
            assert call_args is not None
            assert call_args.kwargs["api_base_url"] == "https://api.groq.com/openai/v1"
            assert call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_initialization_with_api_key(self, mock_generator_class) -> None:
        """Test QueryEnhancer initialization with explicit API key."""
        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        enhancer = QueryEnhancer(api_key="test-key")
        # The generator property should be a mock instance
        assert hasattr(enhancer.generator, "run")  # Check it has the expected methods
        # Check that the generator was initialized with the correct parameters
        mock_generator_class.assert_called_once()
        call_args = mock_generator_class.call_args
        assert call_args is not None
        assert call_args.kwargs["api_base_url"] == "https://api.groq.com/openai/v1"
        assert call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_multi_queries(self, mock_generator_class) -> None:
        """Test multi-query generation."""
        mock_generator = MagicMock()
        mock_generator.run.return_value = {
            "replies": [MagicMock(text="Alternative query 1\nAlternative query 2")]
        }
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            queries = enhancer.generate_multi_queries("original query", num_queries=3)

            assert len(queries) <= 3
            assert "original query" in queries

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_multi_queries_invalid_count(self, mock_generator_class) -> None:
        """Test error handling for invalid query count."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            with pytest.raises(ValueError, match="num_queries must be >= 1"):
                enhancer.generate_multi_queries("query", num_queries=0)

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_hypothetical_documents(self, mock_generator_class) -> None:
        """Test hypothetical document generation."""
        mock_generator = MagicMock()
        mock_generator.run.return_value = {
            "replies": [MagicMock(text="Doc 1 --- Doc 2")]
        }
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            docs = enhancer.generate_hypothetical_documents("query", num_docs=2)

            assert isinstance(docs, list)
            assert len(docs) <= 2

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_hypothetical_documents_invalid_count(
        self, mock_generator_class
    ) -> None:
        """Test error handling for invalid document count."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            with pytest.raises(ValueError, match="num_docs must be >= 1"):
                enhancer.generate_hypothetical_documents("query", num_docs=0)

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_enhance_query_multi_query(self, mock_generator_class) -> None:
        """Test enhance_query with multi_query type."""
        mock_generator = MagicMock()
        mock_generator.run.return_value = {
            "replies": [MagicMock(text="Query 2\nQuery 3")]
        }
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            result = enhancer.enhance_query("query", enhancement_type="multi_query")

            assert isinstance(result, list)

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_enhance_query_invalid_type(self, mock_generator_class) -> None:
        """Test error handling for invalid enhancement type."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            with pytest.raises(ValueError, match="Unsupported enhancement type"):
                enhancer.enhance_query("query", enhancement_type="invalid")

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_step_back_query(self, mock_generator_class) -> None:
        """Test step-back query generation."""
        mock_generator_instance = MagicMock()

        # Create a mock reply object with text attribute
        reply_mock = MagicMock()
        reply_mock.text = "Abstracted question"

        # Mock the run method to return the expected structure
        mock_generator_instance.run.return_value = {"replies": [reply_mock]}
        mock_generator_class.return_value = mock_generator_instance

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            step_back = enhancer.generate_step_back_query("Specific question")

            assert step_back == "Abstracted question"

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_enhance_query_hyde(self, mock_generator_class) -> None:
        """Test enhance_query with hyde type."""
        mock_generator_instance = MagicMock()
        reply_mock = MagicMock()
        reply_mock.text = "Document 1 --- Document 2"
        mock_generator_instance.run.return_value = {"replies": [reply_mock]}
        mock_generator_class.return_value = mock_generator_instance

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            result = enhancer.enhance_query("query", enhancement_type="hyde")

            assert isinstance(result, list)
            assert len(result) >= 1

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_enhance_query_step_back(self, mock_generator_class) -> None:
        """Test enhance_query with step_back type."""
        mock_generator = MagicMock()
        mock_generator.run.return_value = {
            "replies": [MagicMock(text="Abstracted question")]
        }
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            result = enhancer.enhance_query("query", enhancement_type="step_back")

            assert isinstance(result, list)
            assert len(result) == 2  # original + step-back

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_missing_api_key_error(self, mock_generator_class) -> None:
        """Test that ValueError is raised when no API key is provided."""
        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="GROQ_API_KEY required"),
        ):
            QueryEnhancer()

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_multi_queries_error_handling(self, mock_generator_class) -> None:
        """Test error handling in generate_multi_queries."""
        mock_generator = MagicMock()
        mock_generator.run.side_effect = Exception("API Error")
        mock_generator_class.return_value = mock_generator

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            result = enhancer.generate_multi_queries("query", num_queries=3)

            # Should return the original query as fallback
            assert result == ["query"]

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_hypothetical_documents_error_handling(
        self, mock_generator_class
    ) -> None:
        """Test error handling in generate_hypothetical_documents."""
        mock_generator_instance = MagicMock()
        mock_generator_instance.run.side_effect = Exception("API Error")
        mock_generator_class.return_value = mock_generator_instance

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            result = enhancer.generate_hypothetical_documents("query", num_docs=2)

            # Should return the original query as fallback
            assert result == ["query"]

    @patch("vectordb.haystack.components.query_enhancer.OpenAIChatGenerator")
    def test_generate_step_back_query_error_handling(
        self, mock_generator_class
    ) -> None:
        """Test error handling in generate_step_back_query."""
        mock_generator_instance = MagicMock()
        mock_generator_instance.run.side_effect = Exception("API Error")
        mock_generator_class.return_value = mock_generator_instance

        from vectordb.haystack.components.query_enhancer import QueryEnhancer

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            enhancer = QueryEnhancer()
            result = enhancer.generate_step_back_query("query")

            # Should return the original query as fallback
            assert result == "query"
