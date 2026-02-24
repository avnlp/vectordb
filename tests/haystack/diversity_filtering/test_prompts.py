"""Tests for prompt utilities."""

import pytest

from vectordb.haystack.diversity_filtering.utils.prompts import (
    format_documents,
    get_prompt_template,
)


class TestPrompts:
    """Test prompt templates and formatting."""

    def test_get_prompt_template_triviaqa(self):
        """Test getting TriviaQA prompt template."""
        template = get_prompt_template("triviaqa")
        assert "Question:" in template
        assert "{query}" in template
        assert "{documents}" in template

    def test_get_prompt_template_arc(self):
        """Test getting ARC prompt template."""
        template = get_prompt_template("arc")
        assert "multiple-choice" in template or "question" in template.lower()
        assert "{query}" in template

    def test_get_prompt_template_popqa(self):
        """Test getting PopQA prompt template."""
        template = get_prompt_template("popqa")
        assert "{query}" in template
        assert "{documents}" in template

    def test_get_prompt_template_factscore(self):
        """Test getting FactScore prompt template."""
        template = get_prompt_template("factscore")
        assert "fact" in template.lower() or "statement" in template.lower()
        assert "{query}" in template

    def test_get_prompt_template_earnings_calls(self):
        """Test getting earnings calls prompt template."""
        template = get_prompt_template("earnings_calls")
        assert "{query}" in template
        assert "{documents}" in template

    def test_get_prompt_template_invalid(self):
        """Test error for invalid dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_prompt_template("invalid_dataset")

    def test_format_documents_with_content(self):
        """Test formatting documents with 'content' field."""
        docs = [
            {"content": "Document 1 content"},
            {"content": "Document 2 content"},
        ]

        formatted = format_documents(docs)

        assert "Document 1:" in formatted
        assert "Document 1 content" in formatted
        assert "Document 2:" in formatted
        assert "Document 2 content" in formatted

    def test_format_documents_with_text(self):
        """Test formatting documents with 'text' field."""
        docs = [
            {"text": "Document 1 text"},
            {"text": "Document 2 text"},
        ]

        formatted = format_documents(docs)

        assert "Document 1:" in formatted
        assert "Document 1 text" in formatted
        assert "Document 2:" in formatted
        assert "Document 2 text" in formatted

    def test_format_documents_empty(self):
        """Test formatting empty document list."""
        formatted = format_documents([])
        assert formatted == ""

    def test_format_documents_with_metadata(self):
        """Test formatting documents with metadata."""
        docs = [
            {
                "content": "Main content",
                "metadata": {"source": "wikipedia", "score": 0.95},
            }
        ]

        formatted = format_documents(docs)

        assert "Main content" in formatted
        assert "Document 1:" in formatted

    def test_format_documents_single(self):
        """Test formatting single document."""
        docs = [{"content": "Single document"}]

        formatted = format_documents(docs)

        assert "Document 1: Single document" in formatted
        # Should not have Document 2
        assert "Document 2:" not in formatted

    def test_format_documents_order(self):
        """Test documents are numbered in order."""
        docs = [{"content": f"Document {i} content"} for i in range(1, 6)]

        formatted = format_documents(docs)

        # Check numbering
        for i in range(1, 6):
            assert f"Document {i}:" in formatted
