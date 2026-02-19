"""Shared fixtures for vector database wrapper tests.

This module provides pytest fixtures specifically designed for testing
vector database wrapper implementations. These fixtures create mock data
structures and components that simulate database interactions without
requiring actual database connections.

Fixtures:
    sample_documents: Haystack Document objects with embeddings for testing.
    query_embedding: Sample 384-dimensional query embedding vector.
    mock_client: Mock vector database client for unit testing.
    mock_collection: Mock collection object with count method.
    base_db_config: Base configuration dictionary for database tests.

Note:
    Sample documents use 384-dimensional embeddings to match the
    sentence-transformers/all-MiniLM-L6-v2 model output dimensions.
    This ensures consistency across all database wrapper tests.
"""

from unittest.mock import MagicMock

import pytest
from haystack import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for database testing."""
    return [
        Document(
            content="Machine learning fundamentals",
            meta={"id": "doc1", "source": "textbook", "chapter": 1},
            embedding=[0.1, 0.2, 0.3] * 128,
        ),
        Document(
            content="Deep learning architectures",
            meta={"id": "doc2", "source": "paper", "chapter": 2},
            embedding=[0.4, 0.5, 0.6] * 128,
        ),
        Document(
            content="Natural language processing",
            meta={"id": "doc3", "source": "blog", "chapter": 3},
            embedding=[0.7, 0.8, 0.9] * 128,
        ),
    ]


@pytest.fixture
def query_embedding() -> list[float]:
    """Create a sample query embedding."""
    return [0.15, 0.25, 0.35] * 128


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock vector database client."""
    return MagicMock()


@pytest.fixture
def mock_collection() -> MagicMock:
    """Create a mock collection object."""
    collection = MagicMock()
    collection.name = "test_collection"
    collection.count = MagicMock(return_value=3)
    return collection


@pytest.fixture
def base_db_config() -> dict:
    """Create base database configuration."""
    return {
        "collection_name": "test_collection",
        "dimension": 384,
        "metric": "cosine",
        "recreate": False,
    }
