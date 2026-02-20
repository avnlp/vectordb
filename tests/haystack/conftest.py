"""Shared fixtures for Haystack tests.

This module provides pytest fixtures designed for testing Haystack
pipeline integrations with vector databases. These fixtures simulate
Haystack components and provide test configuration templates.

Fixtures:
    sample_documents: Haystack Document objects for testing retrieval.
    sample_embedding: Sample 384-dimensional embedding vector.
    mock_dataloader: Mock dataloader returning sample documents.
    mock_document_embedder: Mock embedder for document encoding.
    mock_text_embedder: Mock embedder for query encoding.
    base_config: Base configuration template for tests.
    milvus_config: Milvus-specific configuration.
    qdrant_config: Qdrant-specific configuration.
    pinecone_config: Pinecone-specific configuration.
    weaviate_config: Weaviate-specific configuration.
    chroma_config: Chroma-specific configuration.

Note:
    Sample documents use 384-dimensional embeddings to match
    sentence-transformers/all-MiniLM-L6-v2 output dimensions.
"""

from unittest.mock import MagicMock

import pytest
from haystack import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            meta={"id": "1", "source": "wiki"},
            embedding=[0.1] * 384,
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers.",
            meta={"id": "2", "source": "paper"},
            embedding=[0.2] * 384,
        ),
        Document(
            content="Natural language processing powers chatbots and translation.",
            meta={"id": "3", "source": "blog"},
            embedding=[0.15] * 384,
        ),
    ]


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.15] * 384


@pytest.fixture
def mock_dataloader(sample_documents: list[Document]) -> MagicMock:
    """Create a mock dataloader."""
    loader = MagicMock()
    loader.get_documents.return_value = sample_documents
    return loader


@pytest.fixture
def mock_document_embedder(sample_documents: list[Document]) -> MagicMock:
    """Create a mock document embedder."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    embedded_docs = [
        Document(
            content=doc.content,
            meta=doc.meta,
            embedding=[0.1 * (i + 1)] * 384,
        )
        for i, doc in enumerate(sample_documents)
    ]
    embedder.run = MagicMock(return_value={"documents": embedded_docs})
    return embedder


@pytest.fixture
def mock_text_embedder(sample_embedding: list[float]) -> MagicMock:
    """Create a mock text embedder for queries."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    embedder.run = MagicMock(return_value={"embedding": sample_embedding})
    return embedder


@pytest.fixture
def base_config() -> dict:
    """Create a base test configuration."""
    return {
        "dataloader": {"type": "arc", "split": "test", "limit": 10},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "search": {"top_k": 5},
        "semantic_diversification": {"enabled": False},
        "rag": {"enabled": False},
    }


@pytest.fixture
def milvus_config(base_config: dict) -> dict:
    """Create Milvus-specific test config."""
    return {
        **base_config,
        "milvus": {
            "uri": "http://localhost:19530",
            "token": "",
            "collection_name": "test_collection",
            "dimension": 384,
            "metric": "cosine",
            "recreate": False,
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
            "collection_name": "test_collection",
            "dimension": 384,
            "recreate": False,
        },
    }


@pytest.fixture
def pinecone_config(base_config: dict) -> dict:
    """Create Pinecone-specific test config."""
    return {
        **base_config,
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-index",
            "namespace": "test",
            "dimension": 384,
            "metric": "cosine",
            "recreate": False,
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
            "class_name": "TestClass",
            "recreate": False,
        },
    }


@pytest.fixture
def chroma_config(base_config: dict) -> dict:
    """Create Chroma-specific test config."""
    return {
        **base_config,
        "chroma": {
            "host": "localhost",
            "port": 8000,
            "collection_name": "test_collection",
            "recreate": False,
        },
    }
