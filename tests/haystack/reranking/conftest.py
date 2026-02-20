"""Shared fixtures for reranking pipeline tests.

This module provides pytest fixtures for testing reranking implementations
in Haystack. Reranking pipelines apply cross-encoder or LLM-based models
to improve retrieval precision by rescoring initial search results.

Fixtures:
    sample_documents: Session-scoped Haystack Document objects covering
        AI and machine learning topics for reranking evaluation.
    sample_query: Sample search query for testing reranker relevance scoring.

Note:
    Fixtures use session scope to avoid recreating documents across tests,
    improving test suite performance for reranking pipeline validation.
"""

import pytest
from haystack import Document


@pytest.fixture(scope="session")
def sample_documents() -> list[Document]:
    """Create sample documents for reranking tests.

    Returns:
        List of Haystack Documents covering AI-related topics
        for evaluating reranker relevance scoring.
    """
    return [
        Document(content="This is the first document about AI."),
        Document(content="This is the second document about machine learning."),
        Document(content="This is the third document about neural networks."),
        Document(content="This is the fourth document about deep learning."),
        Document(content="This is the fifth document about transformers."),
    ]


@pytest.fixture(scope="session")
def sample_query() -> str:
    """Create a sample search query for reranking tests.

    Returns:
        Query string about artificial intelligence for testing
        reranker relevance scoring against sample documents.
    """
    return "What is artificial intelligence?"
