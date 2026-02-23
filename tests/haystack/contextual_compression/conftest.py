"""Shared fixtures for contextual compression pipeline tests.

This module provides pytest fixtures for testing contextual compression
implementations in Haystack. Contextual compression reduces token usage
and noise by summarizing or filtering retrieved documents before LLM generation.

Fixtures:
    sample_documents: Haystack Documents with embeddings and relevance scores.
    sample_embedding: Sample 384-dimensional query embedding vector.
    mock_text_embedder: Mocked text embedder for query vectorization.
    mock_compressor: Mocked compressor component for reranking tests.
    base_config: Base configuration with embedding and compression settings.
    pinecone_config: Pinecone-specific compression pipeline configuration.
    milvus_config: Milvus-specific compression pipeline configuration.
    qdrant_config: Qdrant-specific compression pipeline configuration.
    weaviate_config: Weaviate-specific compression pipeline configuration.
    chroma_config: Chroma-specific compression pipeline configuration.

Note:
    The rerankers module is mocked at import time to avoid external
    dependencies during unit testing of compression pipelines.
"""

import sys
from unittest.mock import MagicMock

import pytest
from haystack import Document


# Mock rerankers module to avoid external dependencies during testing.
mock_rerankers = MagicMock()
mock_rerankers.UnifiedReranker = MagicMock()
sys.modules["vectordb.haystack.components.rerankers"] = mock_rerankers


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents with embeddings and scores for testing.

    Returns:
        List of Haystack Documents with content, metadata including
        relevance scores, and 384-dimensional embeddings.
    """
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            meta={"id": "1", "source": "wiki", "category": "ml", "score": 0.95},
            embedding=[0.1] * 384,
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers.",
            meta={"id": "2", "source": "paper", "category": "dl", "score": 0.90},
            embedding=[0.2] * 384,
        ),
        Document(
            content="Natural language processing powers chatbots and translation.",
            meta={"id": "3", "source": "blog", "category": "nlp", "score": 0.85},
            embedding=[0.15] * 384,
        ),
    ]


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.15] * 384


@pytest.fixture
def mock_text_embedder(sample_embedding: list[float]) -> MagicMock:
    """Create a mock text embedder for queries."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    embedder.run = MagicMock(return_value={"embedding": sample_embedding})
    return embedder


@pytest.fixture
def mock_compressor(sample_documents: list[Document]) -> MagicMock:
    """Create a mock compressor for reranking."""
    compressor = MagicMock()
    compressor.run = MagicMock(return_value={"documents": sample_documents[:2]})
    return compressor


@pytest.fixture
def base_config() -> dict:
    """Create a base test configuration."""
    return {
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
            "dimension": 384,
        },
        "retrieval": {"top_k": 10},
        "compression": {
            "type": "reranking",
            "reranker": {
                "type": "cross_encoder",
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_k": 5,
            },
        },
    }


@pytest.fixture
def pinecone_config(base_config: dict) -> dict:
    """Create Pinecone-specific test config."""
    return {
        **base_config,
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-compression-index",
            "namespace": "default",
            "dimension": 384,
            "metric": "cosine",
        },
    }


@pytest.fixture
def milvus_config(base_config: dict) -> dict:
    """Create Milvus-specific test config."""
    return {
        **base_config,
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_compression",
            "dimension": 384,
            "metric": "cosine",
        },
    }


@pytest.fixture
def qdrant_config(base_config: dict) -> dict:
    """Create Qdrant-specific test config."""
    return {
        **base_config,
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_compression",
            "dimension": 384,
        },
    }


@pytest.fixture
def weaviate_config(base_config: dict) -> dict:
    """Create Weaviate-specific test config."""
    return {
        **base_config,
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestCompression",
        },
    }


@pytest.fixture
def chroma_config(base_config: dict) -> dict:
    """Create Chroma-specific test config."""
    return {
        **base_config,
        "chroma": {
            "path": "/tmp/test_chroma_compression",
            "collection_name": "test_compression",
        },
    }
