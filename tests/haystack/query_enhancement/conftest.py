"""Shared fixtures for query enhancement pipeline tests.

This module provides pytest fixtures for testing query enhancement
implementations in Haystack. Query enhancement improves retrieval by
transforming queries using techniques like multi-query generation,
HyDE (Hypothetical Document Embeddings), and query decomposition.

Fixtures:
    mock_document: Mocked Haystack Document with content and embedding.
    sample_config: Configuration dictionary with query enhancement settings
        including multi-query, HyDE parameters, and LLM configuration.

Note:
    Query enhancement tests validate that transformed queries improve
    retrieval recall by generating semantically diverse query variants
    or hypothetical answer documents for embedding-based search.
"""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_document() -> Mock:
    """Create a mock Haystack Document for testing.

    Returns:
        Mock Document object with content, metadata, and embedding
        attributes for testing query enhancement pipeline outputs.
    """
    doc = Mock()
    doc.content = "test content"
    doc.meta = {}
    doc.embedding = [0.1, 0.2, 0.3]
    return doc


@pytest.fixture
def sample_config() -> dict:
    """Create sample configuration for query enhancement tests.

    Returns:
        Configuration dictionary with dataloader, embeddings, query
        enhancement, database, and RAG settings for pipeline testing.
    """
    return {
        "dataloader": {"type": "test", "params": {"limit": 10}},
        "embeddings": {"model": "all-MiniLM-L6-v2", "params": {}},
        "query_enhancement": {
            "type": "multi_query",
            "num_queries": 3,
            "num_hyde_docs": 3,
            "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 10,
        },
        "pinecone": {
            "api_key": "test-api-key",
            "index_name": "test-index",
            "namespace": "test-namespace",
        },
        "logging": {"level": "INFO", "name": "test"},
        "rag": {"enabled": False},
    }
