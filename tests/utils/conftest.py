"""Shared fixtures for utility tests.

This module provides pytest fixtures used across utility tests,
particularly for testing document converters and evaluation utilities.

Fixtures:
    sample_documents: Haystack Document objects for converter testing.
    mock_llm: Mock LLM object for testing generation and evaluation.

Note:
    Sample documents use 384-dimensional embeddings matching
    the sentence-transformers/all-MiniLM-L6-v2 model output.
"""

from unittest.mock import MagicMock

import pytest
from haystack import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for converter testing."""
    return [
        Document(
            content="Test document 1.",
            meta={"id": "1", "source": "test"},
            embedding=[0.1] * 384,
        ),
        Document(
            content="Test document 2.",
            meta={"id": "2", "source": "test"},
            embedding=[0.2] * 384,
        ),
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.generate = MagicMock(return_value="Generated response")
    return llm
