"""Tests for RAG (Retrieval-Augmented Generation) utilities.

This module tests the RAGHelper class which provides utilities for building
RAG pipelines with LangChain. RAG combines document retrieval with LLM
generation to produce grounded, factual answers.

RAGHelper Methods:
    create_llm: Factory method for creating ChatGroq LLM instances
    format_prompt: Formats retrieved documents into a prompt template
    generate: Invokes LLM with formatted context to produce answers

Test Classes:
    TestCreateLLM: LLM factory with config and environment variable support
    TestFormatPrompt: Prompt formatting with document numbering and templates
    TestGenerate: End-to-end generation with LLM invocation
    TestRAGHelperEdgeCases: Unicode, special characters, and boundary conditions

Configuration:
    RAG is configured via dict with keys: enabled, api_key, model, temperature,
    max_tokens. API key can also be provided via GROQ_API_KEY environment variable.

All tests mock ChatGroq to avoid external API calls.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from vectordb.langchain.utils.rag import RAGHelper


class TestCreateLLM:
    """Tests for RAGHelper.create_llm LLM factory method.

    Validates creation of ChatGroq instances from configuration dicts.
    Supports both explicit API keys and environment variable fallback.

    Configuration Keys:
        enabled: Boolean to enable/disable RAG (default: False)
        api_key: Groq API key (optional, falls back to GROQ_API_KEY env var)
        model: Model name (default: llama-3.3-70b-versatile)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum output tokens (default: 2048)
    """

    def test_create_llm_disabled(self):
        """Test create_llm returns None when RAG is disabled."""
        config = {"rag": {"enabled": False}}
        result = RAGHelper.create_llm(config)
        assert result is None

    def test_create_llm_defaults_enabled(self):
        """Test create_llm returns None when rag key exists but enabled not set."""
        config = {"rag": {}}
        result = RAGHelper.create_llm(config)
        assert result is None

    @patch.dict(os.environ, {}, clear=True)
    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_enabled_no_api_key(self, mock_chat_groq):
        """Test create_llm creates ChatGroq with None api_key when not provided."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        config = {"rag": {"enabled": True}}
        result = RAGHelper.create_llm(config)

        # ChatGroq is called with api_key=None
        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["api_key"] is None
        assert result == mock_groq

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_with_api_key(self, mock_chat_groq):
        """Test create_llm creates ChatGroq instance with API key."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        config = {
            "rag": {
                "enabled": True,
                "api_key": "test-api-key",
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.5,
                "max_tokens": 1000,
            }
        }
        result = RAGHelper.create_llm(config)

        mock_chat_groq.assert_called_once_with(
            model="llama-3.3-70b-versatile",
            api_key="test-api-key",
            temperature=0.5,
            max_tokens=1000,
        )
        assert result == mock_groq

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_with_env_api_key(self, mock_chat_groq):
        """Test create_llm uses environment variable for API key."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        with patch.dict(os.environ, {"GROQ_API_KEY": "env-api-key"}):
            config = {
                "rag": {
                    "enabled": True,
                    "model": "llama-3.3-70b-versatile",
                }
            }
            RAGHelper.create_llm(config)

            mock_chat_groq.assert_called_once()
            call_kwargs = mock_chat_groq.call_args[1]
            assert call_kwargs["api_key"] == "env-api-key"

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_default_model(self, mock_chat_groq):
        """Test create_llm uses default model when not specified."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        config = {
            "rag": {
                "enabled": True,
                "api_key": "test-key",
            }
        }
        RAGHelper.create_llm(config)

        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_default_temperature(self, mock_chat_groq):
        """Test create_llm uses default temperature when not specified."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        config = {
            "rag": {
                "enabled": True,
                "api_key": "test-key",
            }
        }
        RAGHelper.create_llm(config)

        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_default_max_tokens(self, mock_chat_groq):
        """Test create_llm uses default max_tokens when not specified."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        config = {
            "rag": {
                "enabled": True,
                "api_key": "test-key",
            }
        }
        RAGHelper.create_llm(config)

        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["max_tokens"] == 2048


class TestFormatPrompt:
    """Tests for RAGHelper.format_prompt prompt template formatting.

    Validates conversion of retrieved documents into formatted prompt strings.
    Documents are numbered and concatenated into a context block, then combined
    with the user query using a template.

    Template Placeholders:
        {context}: Formatted document content with numbering
        {query}: Original user query

    Default Behavior:
        Uses DEFAULT_PROMPT_TEMPLATE with numbered documents and "Question:/Answer:"
        format suitable for most QA use cases.
    """

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Document 1 content", metadata={"source": "doc1"}),
            Document(page_content="Document 2 content", metadata={"source": "doc2"}),
            Document(page_content="Document 3 content", metadata={"source": "doc3"}),
        ]

    def test_default_template(self, sample_documents):
        """Test format_prompt uses default template."""
        result = RAGHelper.format_prompt("test query", sample_documents)

        assert "Document 1 content" in result
        assert "Document 2 content" in result
        assert "Document 3 content" in result
        assert "Question: test query" in result
        assert "Answer:" in result

    def test_custom_template(self, sample_documents):
        """Test format_prompt with custom template."""
        template = "Context: {context}\n\nQuery: {query}\n\nResponse:"
        result = RAGHelper.format_prompt(
            "test query", sample_documents, template=template
        )

        assert "Context:" in result
        assert "Query: test query" in result
        assert "Response:" in result

    def test_empty_documents(self):
        """Test format_prompt with no documents."""
        result = RAGHelper.format_prompt("test query", [])

        assert "Question: test query" in result
        assert "Answer:" in result
        # Context should be empty
        assert result.startswith("\n\n")

    def test_single_document(self):
        """Test format_prompt with single document."""
        docs = [Document(page_content="Single doc content", metadata={})]
        result = RAGHelper.format_prompt("test query", docs)

        assert "Single doc content" in result
        assert "Document 1:" in result  # Still numbered

    def test_document_numbering(self, sample_documents):
        """Test that documents are properly numbered."""
        result = RAGHelper.format_prompt("test query", sample_documents)

        assert "Document 1:" in result
        assert "Document 2:" in result
        assert "Document 3:" in result

    def test_preserves_document_content(self, sample_documents):
        """Test that full document content is preserved."""
        long_content = "This is a very long document content " * 10
        docs = [Document(page_content=long_content, metadata={})]
        result = RAGHelper.format_prompt("test query", docs)

        assert long_content in result

    def test_template_with_custom_formatting(self):
        """Test template with custom formatting placeholders."""
        template = "CONTEXT: {context}\nQUERY: {query}\n---"
        docs = [Document(page_content="Relevant info", metadata={})]
        result = RAGHelper.format_prompt("my question", docs, template=template)

        assert "CONTEXT:" in result
        assert "QUERY: my question" in result
        assert "---" in result


class TestGenerate:
    """Tests for RAGHelper.generate end-to-end answer generation.

    Validates the complete RAG generation pipeline: format prompt, invoke LLM,
    and return generated answer. The generate method orchestrates prompt
    formatting and LLM invocation into a single call.

    Pipeline Steps:
        1. Format documents and query into prompt via format_prompt
        2. Invoke LLM with formatted prompt
        3. Extract and return content from LLM response

    Return Value:
        String containing LLM response content (not stripped).
    """

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Document 1 content", metadata={"source": "doc1"}),
            Document(page_content="Document 2 content", metadata={"source": "doc2"}),
        ]

    @patch("vectordb.langchain.utils.rag.RAGHelper.format_prompt")
    def test_generate_calls_llm(self, mock_format_prompt, sample_documents):
        """Test generate calls LLM with formatted prompt."""
        mock_format_prompt.return_value = "formatted prompt"

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Generated answer"
        mock_llm.invoke.return_value = mock_response

        result = RAGHelper.generate(
            llm=mock_llm,
            query="test query",
            documents=sample_documents,
        )

        mock_llm.invoke.assert_called_once_with("formatted prompt")
        assert result == "Generated answer"

    @patch("vectordb.langchain.utils.rag.RAGHelper.format_prompt")
    def test_generate_with_custom_template(self, mock_format_prompt, sample_documents):
        """Test generate with custom prompt template."""
        mock_format_prompt.return_value = "custom formatted prompt"

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Answer with custom template"
        mock_llm.invoke.return_value = mock_response

        template = "Custom: {context} | Query: {query}"
        RAGHelper.generate(
            llm=mock_llm,
            query="test query",
            documents=sample_documents,
            template=template,
        )

        mock_format_prompt.assert_called_once_with(
            "test query", sample_documents, template
        )

    @patch("vectordb.langchain.utils.rag.RAGHelper.format_prompt")
    def test_generate_empty_documents(self, mock_format_prompt, sample_documents):
        """Test generate with no documents."""
        mock_format_prompt.return_value = "prompt with no context"

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Answer without context"
        mock_llm.invoke.return_value = mock_response

        result = RAGHelper.generate(
            llm=mock_llm,
            query="test query",
            documents=[],
        )

        mock_llm.invoke.assert_called_once()
        assert result == "Answer without context"

    @patch("vectordb.langchain.utils.rag.RAGHelper.format_prompt")
    def test_generate_preserves_response_content(
        self, mock_format_prompt, sample_documents
    ):
        """Test that generate returns response.content."""
        mock_format_prompt.return_value = "prompt"

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "  Answer with whitespace  "
        mock_llm.invoke.return_value = mock_response

        result = RAGHelper.generate(
            llm=mock_llm,
            query="test query",
            documents=sample_documents,
        )

        assert result == "  Answer with whitespace  "


class TestRAGHelperEdgeCases:
    """Edge case tests for RAGHelper boundary conditions.

    Validates robust handling of unusual inputs and configurations.
    RAGHelper should gracefully handle empty configs, missing keys,
    unicode content, and extreme parameter values.

    Edge Cases Covered:
        - Empty or missing configuration sections
        - Temperature extremes (0.0 and 1.5+)
        - Unicode and special characters in documents
        - Very long queries and many documents
        - Missing API keys with environment fallback
    """

    def test_default_prompt_template_constant(self):
        """Test that DEFAULT_PROMPT_TEMPLATE is properly defined."""
        assert "{context}" in RAGHelper.DEFAULT_PROMPT_TEMPLATE
        assert "{query}" in RAGHelper.DEFAULT_PROMPT_TEMPLATE

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_with_empty_config(self, mock_chat_groq):
        """Test create_llm with empty config dict."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            config = {}
            result = RAGHelper.create_llm(config)

            # Should return None since rag config is missing
            assert result is None

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_with_zero_temperature(self, mock_chat_groq):
        """Test create_llm with temperature set to 0."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        config = {
            "rag": {
                "enabled": True,
                "api_key": "test-key",
                "temperature": 0.0,
            }
        }
        RAGHelper.create_llm(config)

        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    @patch("vectordb.langchain.utils.rag.ChatGroq")
    def test_create_llm_with_high_temperature(self, mock_chat_groq):
        """Test create_llm with high temperature."""
        mock_groq = MagicMock()
        mock_chat_groq.return_value = mock_groq

        config = {
            "rag": {
                "enabled": True,
                "api_key": "test-key",
                "temperature": 1.5,
            }
        }
        RAGHelper.create_llm(config)

        call_kwargs = mock_chat_groq.call_args[1]
        assert call_kwargs["temperature"] == 1.5

    def test_format_prompt_with_unicode(self):
        """Test format_prompt with unicode characters."""
        docs = [
            Document(page_content="Unicode: 你好世界 🌍", metadata={}),
        ]
        result = RAGHelper.format_prompt("unicode query?", docs)

        assert "你好世界" in result
        assert "🌍" in result

    def test_format_prompt_with_special_characters(self):
        """Test format_prompt with special characters in documents."""
        docs = [
            Document(page_content="Special: <>&\"'\n\t", metadata={}),
        ]
        result = RAGHelper.format_prompt("special query", docs)

        assert "<>" in result or "Special:" in result

    @patch("vectordb.langchain.utils.rag.RAGHelper.format_prompt")
    def test_generate_with_long_query(self, mock_format_prompt, sample_documents):
        """Test generate with a very long query."""
        mock_format_prompt.return_value = "formatted"

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Answer"
        mock_llm.invoke.return_value = mock_response

        long_query = "query " * 1000
        result = RAGHelper.generate(
            llm=mock_llm,
            query=long_query,
            documents=sample_documents,
        )

        assert result == "Answer"

    def test_format_prompt_many_documents(self):
        """Test format_prompt with many documents."""
        docs = [
            Document(page_content=f"Document {i} content", metadata={})
            for i in range(20)
        ]
        result = RAGHelper.format_prompt("test query", docs)

        assert "Document 1:" in result
        assert "Document 20:" in result
