"""Shared fixtures for metadata filtering pipeline tests.

This module provides pytest fixtures for testing metadata filtering
implementations in Haystack. Metadata filtering enables precise document
retrieval by combining vector similarity with attribute-based constraints.

Fixtures:
    sample_documents: Haystack Documents with category metadata and embeddings.
    sample_embedding: Sample 384-dimensional query embedding vector.
    mock_document_embedder: Mocked embedder for document vectorization.
    mock_text_embedder: Mocked embedder for query vectorization.
    base_config: Base configuration with embedding and filter settings.
    pinecone_config: Pinecone-specific metadata filtering configuration.
    milvus_config: Milvus-specific metadata filtering configuration.
    qdrant_config: Qdrant-specific metadata filtering configuration.
    weaviate_config: Weaviate-specific metadata filtering configuration.
    chroma_config: Chroma-specific metadata filtering configuration.

Note:
    Metadata filtering tests validate that filter expressions correctly
    combine with vector similarity search to return only documents
    matching both semantic relevance and attribute constraints.
"""

from unittest.mock import MagicMock

import pytest
from haystack import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents with category metadata for filtering tests.

    Returns:
        List of Haystack Documents with source and category metadata
        plus 384-dimensional embeddings for metadata filtering tests.
    """
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            meta={"id": "1", "source": "wiki", "category": "ml"},
            embedding=[0.1] * 384,
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers.",
            meta={"id": "2", "source": "paper", "category": "dl"},
            embedding=[0.2] * 384,
        ),
        Document(
            content="Natural language processing powers chatbots and translation.",
            meta={"id": "3", "source": "blog", "category": "nlp"},
            embedding=[0.15] * 384,
        ),
    ]


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.15] * 384


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
            "dimension": 384,
        },
        "search": {"top_k": 5},
        "metadata_filtering": {
            "test_query": "What is machine learning?",
            "filters": {"field": "category", "operator": "==", "value": "ml"},
        },
    }


@pytest.fixture
def pinecone_config(base_config: dict) -> dict:
    """Create Pinecone-specific test config."""
    return {
        **base_config,
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-metadata-index",
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
            "uri": "http://localhost:19530",
            "token": "",
            "collection_name": "test_metadata",
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
            "collection_name": "test_metadata",
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
            "collection_name": "TestMetadata",
        },
    }


@pytest.fixture
def chroma_config(base_config: dict) -> dict:
    """Create Chroma-specific test config."""
    return {
        **base_config,
        "chroma": {
            "persist_directory": "./test_chroma_data",
            "collection_name": "test_metadata",
        },
    }
