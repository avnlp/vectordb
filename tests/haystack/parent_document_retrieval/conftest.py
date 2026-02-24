"""Shared fixtures for parent document retrieval pipeline tests.

This module provides pytest fixtures for testing parent document retrieval
implementations in Haystack. Parent document retrieval returns complete
parent documents when child chunks match queries, preserving full context.

Fixtures:
    sample_parent_store: InMemoryDocumentStore for parent document storage.
    sample_documents: Haystack Documents with 1024-dimensional embeddings.
    sample_parent_documents: Parent documents with hierarchical metadata.
    sample_leaf_documents: Child documents with parent references.
    sample_embedding: Sample 1024-dimensional query embedding vector.
    mock_document_embedder: Mocked embedder for document vectorization.
    mock_text_embedder: Mocked embedder for query vectorization.
    mock_splitter: Mocked hierarchical document splitter.
    base_config: Base configuration with chunking and retrieval settings.
    pinecone_config: Pinecone-specific parent retrieval configuration.
    milvus_config: Milvus-specific parent retrieval configuration.
    qdrant_config: Qdrant-specific parent retrieval configuration.
    weaviate_config: Weaviate-specific parent retrieval configuration.
    chroma_config: Chroma-specific parent retrieval configuration.

Note:
    Parent document retrieval uses 1024-dimensional embeddings to match
    higher-capacity embedding models used for long-context retrieval.
"""

from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


@pytest.fixture
def sample_parent_store() -> InMemoryDocumentStore:
    """Create a parent document store for testing.

    Returns:
        InMemoryDocumentStore instance for storing parent documents
        during parent-child relationship testing.
    """
    return InMemoryDocumentStore()


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents with embeddings for testing.

    Returns:
        List of Haystack Documents with 1024-dimensional embeddings
        and metadata for testing parent document retrieval pipelines.
    """
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            meta={"id": "1", "source": "wiki"},
            embedding=[0.1] * 1024,
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers.",
            meta={"id": "2", "source": "paper"},
            embedding=[0.2] * 1024,
        ),
        Document(
            content="Natural language processing powers chatbots and translation.",
            meta={"id": "3", "source": "blog"},
            embedding=[0.15] * 1024,
        ),
    ]


@pytest.fixture
def sample_parent_documents() -> list[Document]:
    """Create sample parent documents with hierarchical metadata."""
    return [
        Document(
            content="Machine learning is a subset of AI with many applications.",
            meta={"id": "parent_1", "level": 1, "children_ids": ["child_1", "child_2"]},
        ),
        Document(
            content="Deep learning is a subset of machine learning using neural networks.",
            meta={"id": "parent_2", "level": 1, "children_ids": ["child_3", "child_4"]},
        ),
    ]


@pytest.fixture
def sample_leaf_documents() -> list[Document]:
    """Create sample leaf documents for testing."""
    return [
        Document(
            content="ML is AI",
            meta={"id": "child_1", "parent_id": "parent_1", "level": 2},
            embedding=[0.1] * 1024,
        ),
        Document(
            content="many applications",
            meta={"id": "child_2", "parent_id": "parent_1", "level": 2},
            embedding=[0.12] * 1024,
        ),
        Document(
            content="Deep learning neural",
            meta={"id": "child_3", "parent_id": "parent_2", "level": 2},
            embedding=[0.2] * 1024,
        ),
        Document(
            content="networks subset ML",
            meta={"id": "child_4", "parent_id": "parent_2", "level": 2},
            embedding=[0.22] * 1024,
        ),
    ]


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample embedding vector."""
    return [0.15] * 1024


@pytest.fixture
def mock_document_embedder(sample_leaf_documents: list[Document]) -> MagicMock:
    """Create a mock document embedder."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    embedder.run = MagicMock(return_value={"documents": sample_leaf_documents})
    return embedder


@pytest.fixture
def mock_text_embedder(sample_embedding: list[float]) -> MagicMock:
    """Create a mock text embedder for queries."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    embedder.run = MagicMock(return_value={"embedding": sample_embedding})
    return embedder


@pytest.fixture
def mock_splitter(
    sample_parent_documents: list[Document],
    sample_leaf_documents: list[Document],
) -> MagicMock:
    """Create a mock hierarchical splitter."""
    splitter = MagicMock()
    all_docs = sample_parent_documents + sample_leaf_documents
    splitter.run = MagicMock(return_value={"documents": all_docs})
    return splitter


@pytest.fixture
def base_config() -> dict:
    """Create a base test configuration."""
    return {
        "dataloader": {"type": "arc", "split": "test", "index_limit": 10},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
        },
        "chunking": {
            "parent_chunk_size_words": 100,
            "child_chunk_size_words": 25,
            "split_overlap": 5,
        },
        "retrieval": {"top_k": 5, "merge_threshold": 0.5},
    }


@pytest.fixture
def pinecone_config(base_config: dict) -> dict:
    """Create Pinecone-specific test config."""
    return {
        **base_config,
        "database": {
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-parent-doc-index",
                "namespace": "default",
            },
        },
    }


@pytest.fixture
def milvus_config(base_config: dict) -> dict:
    """Create Milvus-specific test config."""
    return {
        **base_config,
        "database": {
            "milvus": {
                "uri": "http://localhost:19530",
                "token": "",
                "collection_name": "test_parent_doc",
            },
        },
    }


@pytest.fixture
def qdrant_config(base_config: dict) -> dict:
    """Create Qdrant-specific test config."""
    return {
        **base_config,
        "database": {
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_parent_doc",
            },
        },
    }


@pytest.fixture
def weaviate_config(base_config: dict) -> dict:
    """Create Weaviate-specific test config."""
    return {
        **base_config,
        "database": {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDoc",
            },
        },
    }


@pytest.fixture
def chroma_config(base_config: dict) -> dict:
    """Create Chroma-specific test config."""
    return {
        **base_config,
        "database": {
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_parent_doc",
            },
        },
    }
