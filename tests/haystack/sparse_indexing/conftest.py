"""Shared fixtures for sparse indexing and hybrid search tests.

This module provides pytest fixtures specifically designed for testing
sparse indexing and hybrid search implementations. These fixtures provide
mock configurations and data structures for testing BM25, TF-IDF, and
dense-sparse hybrid retrieval methods.

Fixtures:
    mock_config: Complete configuration dictionary for sparse indexing tests.
        Includes settings for:
        - Dataloader configuration (batch size, limits)
        - Sparse model configuration (Splade backend, device settings)
        - Database-specific configurations for all vector stores
        - Indexing parameters (batch size, progress tracking)
        - Query parameters (top-k retrieval)

Note:
    The mock_config uses Splade (Sparse Lexical and Expansion Model)
    as the default sparse encoder, which provides learned sparse
    representations that combine the benefits of BM25 and neural
    expansion.
"""

import pytest


@pytest.fixture
def mock_config():
    """Mock configuration for sparse indexing tests."""
    return {
        "dataloader": {"name": "test_dataloader", "limit": 10, "batch_size": 32},
        "sparse": {
            "model": "prithivida/Splade_PP_en_v2",
            "backend": "torch",
            "device": None,
        },
        "pinecone": {
            "api_key": "test_api_key",
            "index_name": "test_index",
            "namespace": "test_namespace",
        },
        "milvus": {
            "connection_args": {"host": "localhost", "port": "19530"},
            "collection_name": "test_collection",
        },
        "qdrant": {
            "location": "localhost:6333",
            "collection_name": "test_collection",
            "api_key": "test_api_key",
        },
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "test_api_key",
            "index_name": "TestIndex",
        },
        "chroma": {
            "collection_name": "test_collection",
            "persist_directory": "./test_data",
        },
        "indexing": {"batch_size": 100, "show_progress": True},
        "query": {"top_k": 10},
    }
