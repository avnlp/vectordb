"""Shared fixtures for hybrid indexing pipeline tests.

This module provides pytest fixtures for testing hybrid (dense + sparse)
indexing implementations in Haystack. Hybrid indexing combines semantic
embeddings with lexical signals (BM25/Splade) for improved retrieval.

Fixtures:
    sample_documents: Haystack Documents with 384-dimensional dense embeddings.
    sample_documents_with_sparse: Documents with both dense and sparse embeddings.
    sample_embedding: Sample 384-dimensional query embedding vector.
    sample_sparse_embedding: Sample SparseEmbedding for hybrid queries.
    mock_document_embedder: Mocked embedder for document vectorization.
    mock_text_embedder: Mocked embedder for query vectorization.
    mock_sparse_document_embedder: Mocked sparse embedder for documents.
    mock_sparse_text_embedder: Mocked sparse embedder for queries.
    base_config: Base configuration template for hybrid indexing.
    pinecone_config: Pinecone-specific hybrid indexing configuration.
    weaviate_config: Weaviate-specific hybrid indexing configuration.
    chroma_config: Chroma-specific hybrid indexing configuration.
    milvus_config: Milvus-specific hybrid indexing configuration.
    qdrant_config: Qdrant-specific hybrid indexing configuration.

Note:
    Sample embeddings use 384 dimensions to match sentence-transformers/all-MiniLM-L6-v2
    output. Sparse embeddings use compact index-value pairs for efficient storage.
"""

from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents with dense embeddings for testing.

    Returns:
        List of Haystack Documents with 384-dimensional embeddings
        and metadata for testing hybrid indexing pipelines.
    """
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
def sample_documents_with_sparse() -> list[Document]:
    """Create sample documents with both dense and sparse embeddings."""
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            meta={"id": "1", "source": "wiki"},
            embedding=[0.1] * 384,
            sparse_embedding=SparseEmbedding(
                indices=[0, 5, 10], values=[0.5, 0.3, 0.2]
            ),
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers.",
            meta={"id": "2", "source": "paper"},
            embedding=[0.2] * 384,
            sparse_embedding=SparseEmbedding(
                indices=[1, 6, 11], values=[0.4, 0.35, 0.25]
            ),
        ),
        Document(
            content="Natural language processing powers chatbots and translation.",
            meta={"id": "3", "source": "blog"},
            embedding=[0.15] * 384,
            sparse_embedding=SparseEmbedding(
                indices=[2, 7, 12], values=[0.45, 0.32, 0.23]
            ),
        ),
    ]


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.15] * 384


@pytest.fixture
def sample_sparse_embedding() -> SparseEmbedding:
    """Create a sample sparse embedding."""
    return SparseEmbedding(indices=[0, 5, 10, 15], values=[0.5, 0.3, 0.15, 0.05])


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
def mock_sparse_document_embedder(
    sample_documents_with_sparse: list[Document],
) -> MagicMock:
    """Create a mock sparse document embedder."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    embedder.run = MagicMock(return_value={"documents": sample_documents_with_sparse})
    return embedder


@pytest.fixture
def mock_sparse_text_embedder(sample_sparse_embedding: SparseEmbedding) -> MagicMock:
    """Create a mock sparse text embedder."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    embedder.run = MagicMock(return_value={"sparse_embedding": sample_sparse_embedding})
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
        "sparse": {
            "model": "prithivida/Splade_PP_en_v1",
        },
        "search": {"top_k": 5},
    }


@pytest.fixture
def pinecone_config(base_config: dict) -> dict:
    """Create Pinecone-specific test config."""
    return {
        **base_config,
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-hybrid-index",
            "namespace": "default",
            "dimension": 384,
            "metric": "cosine",
            "batch_size": 100,
            "alpha": 0.5,
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
            "collection_name": "TestHybrid",
            "dimension": 384,
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
            "collection_name": "test_hybrid",
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
            "collection_name": "test_hybrid",
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
            "collection_name": "test_hybrid",
            "dimension": 384,
        },
    }
