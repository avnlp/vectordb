"""Tests for RAG generation utilities."""

from unittest.mock import MagicMock, patch

import pytest


class TestCreateRagGenerator:
    """Tests for create_rag_generator function."""

    def test_create_rag_generator_disabled_by_default(self):
        """Test that RAG generator returns None when disabled."""
        from vectordb.haystack.metadata_filtering.common.rag import (
            create_rag_generator,
        )

        config = {}
        result = create_rag_generator(config)
        assert result is None

    def test_create_rag_generator_disabled_explicit(self):
        """Test that RAG generator returns None when explicitly disabled."""
        from vectordb.haystack.metadata_filtering.common.rag import (
            create_rag_generator,
        )

        config = {"rag": {"enabled": False}}
        result = create_rag_generator(config)
        assert result is None

    def test_create_rag_generator_missing_model_raises_error(self):
        """Test that missing model raises ValueError."""
        from vectordb.haystack.metadata_filtering.common.rag import (
            create_rag_generator,
        )

        config = {"rag": {"enabled": True}}
        with pytest.raises(ValueError, match="'rag.model' is required"):
            create_rag_generator(config)

    def test_create_rag_generator_success(self):
        """Test successful RAG generator creation."""
        from vectordb.haystack.metadata_filtering.common.rag import (
            create_rag_generator,
        )

        config = {
            "rag": {
                "enabled": True,
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key",
            }
        }
        with patch(
            "vectordb.haystack.metadata_filtering.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = create_rag_generator(config)

            mock_class.assert_called_once()
            assert result == mock_instance

    def test_create_rag_generator_with_api_base_url(self):
        """Test RAG generator creation with custom API base URL."""
        from vectordb.haystack.metadata_filtering.common.rag import (
            create_rag_generator,
        )

        config = {
            "rag": {
                "enabled": True,
                "model": "llama-3.3-70b-versatile",
                "api_key": "test-key",
                "api_base_url": "https://custom.api.com",
            }
        }
        with patch(
            "vectordb.haystack.metadata_filtering.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = create_rag_generator(config)

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["api_base_url"] == "https://custom.api.com"
            assert result == mock_instance

    def test_create_rag_generator_no_api_key(self):
        """Test RAG generator creation without API key."""
        from vectordb.haystack.metadata_filtering.common.rag import (
            create_rag_generator,
        )

        config = {
            "rag": {
                "enabled": True,
                "model": "llama-3.3-70b-versatile",
            }
        }
        with patch(
            "vectordb.haystack.metadata_filtering.common.rag.OpenAIGenerator"
        ) as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            result = create_rag_generator(config)

            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["api_key"] is None
            assert result == mock_instance


class TestGenerateAnswer:
    """Tests for generate_answer function."""

    def test_generate_answer_empty_documents(self):
        """Test that empty documents returns None."""
        from vectordb.haystack.metadata_filtering.common.rag import generate_answer

        generator = MagicMock()
        result = generate_answer("test query", [], generator)
        assert result is None

    def test_generate_answer_success(self):
        """Test successful answer generation."""
        from haystack import Document

        from vectordb.haystack.metadata_filtering.common.rag import generate_answer

        generator = MagicMock()
        generator.run.return_value = {"replies": ["Generated answer"]}

        documents = [Document(content="Test document content", meta={})]

        result = generate_answer("test query", documents, generator)

        assert result == "Generated answer"
        generator.run.assert_called_once()

    def test_generate_answer_uses_top_k_documents(self):
        """Test that only top-k documents are used."""
        from haystack import Document

        from vectordb.haystack.metadata_filtering.common.rag import generate_answer

        generator = MagicMock()
        generator.run.return_value = {"replies": ["Generated answer"]}

        documents = [Document(content=f"Document {i}", meta={}) for i in range(10)]

        result = generate_answer("test query", documents, generator, max_docs=3)

        assert result == "Generated answer"
        # Check that prompt contains only 3 documents
        call_args = generator.run.call_args
        prompt = call_args[1]["prompt"]
        assert "Document 1" in prompt
        assert "Document 2" in prompt
        assert "Document 3" in prompt
        # Later documents should not be in prompt
        assert "Document 5" not in prompt

    def test_generate_answer_handles_none_content(self):
        """Test handling of documents with None content."""
        from haystack import Document

        from vectordb.haystack.metadata_filtering.common.rag import generate_answer

        generator = MagicMock()
        generator.run.return_value = {"replies": ["Generated answer"]}

        documents = [Document(content=None, meta={"field": "value"})]

        result = generate_answer("test query", documents, generator)

        assert result == "Generated answer"

    def test_generate_answer_no_replies(self):
        """Test handling when generator returns no replies."""
        from haystack import Document

        from vectordb.haystack.metadata_filtering.common.rag import generate_answer

        generator = MagicMock()
        generator.run.return_value = {"replies": []}

        documents = [Document(content="Test", meta={})]

        result = generate_answer("test query", documents, generator)

        assert result is None

    def test_generate_answer_exception(self):
        """Test handling of generator exception."""
        from haystack import Document

        from vectordb.haystack.metadata_filtering.common.rag import generate_answer

        generator = MagicMock()
        generator.run.side_effect = Exception("API error")

        documents = [Document(content="Test", meta={})]

        result = generate_answer("test query", documents, generator)

        assert result is None
