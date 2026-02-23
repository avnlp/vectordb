"""Fixtures for semantic search tests.

This module provides pytest fixtures for testing semantic search pipelines
across all supported vector databases. Includes mock database connections
and embedder simulations.

Mock database fixtures:
    mock_pinecone_db: Simulated PineconeVectorDB instance.
    mock_weaviate_db: Simulated WeaviateVectorDB instance.
    mock_chroma_db: Simulated ChromaVectorDB instance.
    mock_milvus_db: Simulated MilvusVectorDB instance.
    mock_qdrant_db: Simulated QdrantVectorDB instance.

Mock component fixtures:
    mock_embedder: Mock embedder returning 384-dimensional vectors.
        - embed_documents: Returns list of 384-dim vectors.
        - embed_query: Returns single 384-dim query vector.

Note:
    All mock databases return empty query results by default.
    Tests should configure specific return values as needed.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_pinecone_db() -> MagicMock:
    """Create mock PineconeVectorDB."""
    db = MagicMock()
    db.create_index.return_value = None
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_weaviate_db() -> MagicMock:
    """Create mock WeaviateVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_chroma_db() -> MagicMock:
    """Create mock ChromaVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_milvus_db() -> MagicMock:
    """Create mock MilvusVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_qdrant_db() -> MagicMock:
    """Create mock QdrantVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create mock embedder."""
    embedder = MagicMock()
    embedder.embed_documents.return_value = [[0.1] * 384] * 5
    embedder.embed_query.return_value = [0.1] * 384
    return embedder
