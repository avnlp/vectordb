"""Comprehensive tests for Haystack context compression components.

This module tests the ContextCompressor which reduces token usage and noise
by compressing retrieved context before LLM generation.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestContextCompressor:
    """Test suite for ContextCompressor component.

    Tests cover:
    - Initialization with different configurations
    - Abstractive summarization
    - Extractive summarization
    - Relevance filtering
    - Unified compress method with different types
    - Configuration validation
    - Error handling and edge cases
    """

    @pytest.fixture
    def mock_generator(self):
        """Fixture for mocked OpenAIChatGenerator."""
        with patch(
            "vectordb.haystack.components.context_compressor.OpenAIChatGenerator"
        ) as mock:
            mock_gen = MagicMock()
            mock.return_value = mock_gen
            yield mock, mock_gen

    def test_initialization_default(self, mock_generator):
        """Test ContextCompressor initialization with default parameters."""
        mock_cls, mock_gen = mock_generator

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()

        assert compressor is not None
        assert compressor.generator is mock_gen
        mock_cls.assert_called_once()
        call_args = mock_cls.call_args
        assert call_args is not None
        assert call_args.kwargs["api_base_url"] == "https://api.groq.com/openai/v1"
        assert call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_initialization_custom_params(self, mock_generator):
        """Test ContextCompressor initialization with custom parameters."""
        mock_cls, mock_gen = mock_generator

        from vectordb.haystack.components.context_compressor import ContextCompressor

        compressor = ContextCompressor(
            model="llama-3.1-8b-instant",
            api_key="test-api-key",
        )

        assert compressor is not None
        assert compressor.generator is mock_gen
        mock_cls.assert_called_once()
        call_args = mock_cls.call_args
        assert call_args is not None
        assert call_args.kwargs["model"] == "llama-3.1-8b-instant"

    def test_initialization_failure(self, mock_generator):
        """Test ContextCompressor initialization failure handling."""
        mock_cls, _ = mock_generator
        mock_cls.side_effect = Exception("API connection failed")

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
            pytest.raises(Exception, match="API connection failed"),
        ):
            ContextCompressor()

    def test_compress_abstractive_success(self, mock_generator):
        """Test successful abstractive compression."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "This is a compressed summary of the context."
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "This is a very long context with lots of information about machine learning and AI."
        query = "What is machine learning?"

        result = compressor.compress_abstractive(context, query, max_tokens=100)

        assert result == "This is a compressed summary of the context."
        mock_gen.run.assert_called_once()
        call_args = mock_gen.run.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert query in prompt_text
        assert context in prompt_text

    def test_compress_abstractive_default_max_tokens(self, mock_generator):
        """Test abstractive compression with default max_tokens."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Summary"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Some context"
        query = "What is AI?"

        result = compressor.compress_abstractive(context, query)

        assert result == "Summary"
        call_args = mock_gen.run.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "2048" in prompt_text

    def test_compress_abstractive_exception_fallback(self, mock_generator):
        """Test abstractive compression fallback on exception."""
        _, mock_gen = mock_generator
        mock_gen.run.side_effect = Exception("LLM error")

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Original context that should be returned on error"
        query = "What is AI?"

        result = compressor.compress_abstractive(context, query)

        # Should return original context on failure
        assert result == context

    def test_compress_extractive_success(self, mock_generator):
        """Test successful extractive compression."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Sentence one. Sentence two."
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Sentence one. Sentence two. Sentence three. Sentence four."
        query = "What is important?"

        result = compressor.compress_extractive(context, query, num_sentences=2)

        assert result == "Sentence one. Sentence two."
        mock_gen.run.assert_called_once()
        call_args = mock_gen.run.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "2" in prompt_text
        assert query in prompt_text

    def test_compress_extractive_default_num_sentences(self, mock_generator):
        """Test extractive compression with default num_sentences."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Extracted sentences"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Some long context with many sentences."
        query = "What is AI?"

        result = compressor.compress_extractive(context, query)

        assert result == "Extracted sentences"
        call_args = mock_gen.run.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "5" in prompt_text

    def test_compress_extractive_exception_fallback(self, mock_generator):
        """Test extractive compression fallback on exception."""
        _, mock_gen = mock_generator
        mock_gen.run.side_effect = Exception("LLM error")

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Original context"
        query = "What is AI?"

        result = compressor.compress_extractive(context, query)

        assert result == context

    def test_filter_by_relevance_success(self, mock_generator):
        """Test successful relevance filtering."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = (
            "This paragraph is relevant to the query.\n\nThis one is also relevant."
        )
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Relevant paragraph.\n\nIrrelevant paragraph.\n\nAnother relevant paragraph."
        query = "What is relevant?"

        result = compressor.filter_by_relevance(context, query, relevance_threshold=0.6)

        assert (
            result
            == "This paragraph is relevant to the query.\n\nThis one is also relevant."
        )
        mock_gen.run.assert_called_once()
        call_args = mock_gen.run.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "60" in prompt_text  # 0.6 * 100
        assert query in prompt_text

    def test_filter_by_relevance_default_threshold(self, mock_generator):
        """Test relevance filtering with default threshold."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Filtered content"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Some context"
        query = "What is AI?"

        result = compressor.filter_by_relevance(context, query)

        assert result == "Filtered content"
        call_args = mock_gen.run.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "50" in prompt_text  # 0.5 * 100

    def test_filter_by_relevance_exception_fallback(self, mock_generator):
        """Test relevance filtering fallback on exception."""
        _, mock_gen = mock_generator
        mock_gen.run.side_effect = Exception("LLM error")

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Original context"
        query = "What is AI?"

        result = compressor.filter_by_relevance(context, query)

        assert result == context

    def test_compress_abstractive_type(self, mock_generator):
        """Test compress method with abstractive type."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Abstractive summary"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Long context"
        query = "What is AI?"

        result = compressor.compress(
            context, query, compression_type="abstractive", max_tokens=500
        )

        assert result == "Abstractive summary"
        call_args = mock_gen.run.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "500" in prompt_text

    def test_compress_extractive_type(self, mock_generator):
        """Test compress method with extractive type."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Extractive result"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Long context"
        query = "What is AI?"

        result = compressor.compress(
            context, query, compression_type="extractive", num_sentences=3
        )

        assert result == "Extractive result"
        call_args = mock_gen.run.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "3" in prompt_text

    def test_compress_relevance_filter_type(self, mock_generator):
        """Test compress method with relevance_filter type."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Filtered result"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        context = "Long context"
        query = "What is AI?"

        result = compressor.compress(
            context, query, compression_type="relevance_filter", relevance_threshold=0.7
        )

        assert result == "Filtered result"
        call_args = mock_gen.run.call_args
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "70" in prompt_text

    def test_compress_invalid_type(self, mock_generator):
        """Test compress method with invalid compression type."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()

        with pytest.raises(ValueError, match="Unsupported compression type"):
            compressor.compress("context", "query", compression_type="invalid_type")

    def test_compress_with_uppercase_type(self, mock_generator):
        """Test compress method raises error for uppercase type."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()

        # Uppercase type should raise ValueError (compress method doesn't lowercase)
        with pytest.raises(ValueError, match="Unsupported compression type"):
            compressor.compress("context", "query", compression_type="ABSTRACTIVE")

    def test_validate_config_success(self, mock_generator):
        """Test successful configuration validation."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        config = {"type": "abstractive", "max_tokens": 1000}

        result = compressor.validate_config(config)

        assert result == config

    def test_validate_config_default_type(self, mock_generator):
        """Test config validation with default type."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        config = {"max_tokens": 1000}  # No type specified

        result = compressor.validate_config(config)

        assert result == config

    def test_validate_config_invalid_type(self, mock_generator):
        """Test config validation with invalid type."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        config = {"type": "invalid_type"}

        with pytest.raises(ValueError, match="Unsupported compression type"):
            compressor.validate_config(config)

    def test_validate_config_not_dict(self, mock_generator):
        """Test config validation with non-dict input."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()

        with pytest.raises(ValueError, match="Config must be dict"):
            compressor.validate_config("not a dict")

    def test_validate_config_all_valid_types(self, mock_generator):
        """Test config validation with all valid compression types."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()

        valid_types = ["abstractive", "extractive", "relevance_filter"]
        for comp_type in valid_types:
            config = {"type": comp_type}
            result = compressor.validate_config(config)
            assert result == config

    def test_validate_config_uppercase_type(self, mock_generator):
        """Test config validation accepts uppercase type (converts to lowercase)."""
        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()

        # Uppercase type is accepted (converted to lowercase internally)
        config = {"type": "ABSTRACTIVE"}
        result = compressor.validate_config(config)
        assert result == config

    def test_compress_abstractive_empty_context(self, mock_generator):
        """Test abstractive compression with empty context."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = ""
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        result = compressor.compress_abstractive("", "query")

        assert result == ""

    def test_compress_extractive_empty_context(self, mock_generator):
        """Test extractive compression with empty context."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = ""
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        result = compressor.compress_extractive("", "query")

        assert result == ""

    def test_filter_by_relevance_empty_context(self, mock_generator):
        """Test relevance filtering with empty context."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = ""
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        result = compressor.filter_by_relevance("", "query")

        assert result == ""

    def test_compress_whitespace_handling(self, mock_generator):
        """Test that compression returns LLM response as-is with whitespace."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "  Summary with whitespace  \n\n  "
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.context_compressor import ContextCompressor

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            compressor = ContextCompressor()
        result = compressor.compress_abstractive("context", "query")

        # Implementation returns response text as-is without stripping
        assert result == "  Summary with whitespace  \n\n  "
