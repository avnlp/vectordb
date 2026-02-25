"""Shared fixtures for JSON indexing tests.

This module provides pytest fixtures for testing JSON indexing pipelines
across all supported vector databases. Includes base configurations and
database-specific settings for both indexing and search operations.

Indexing configuration fixtures:
    json_indexing_config: Base dataloader and embeddings settings.
    milvus_json_config: Milvus JSON indexing configuration.
    pinecone_json_config: Pinecone JSON indexing configuration.
    qdrant_json_config: Qdrant JSON indexing configuration.
    weaviate_json_config: Weaviate JSON indexing configuration.

Search configuration fixtures:
    json_search_config_with_filters: Base search config with filter conditions.
    milvus_json_search_config: Milvus search with metadata filters.
    pinecone_json_search_config: Pinecone search configuration.
    qdrant_json_search_config: Qdrant search configuration.
    weaviate_json_search_config: Weaviate search configuration.
"""

import pytest


@pytest.fixture
def json_indexing_config() -> dict:
    """Create base JSON indexing configuration for testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
    }


@pytest.fixture
def milvus_json_config(json_indexing_config: dict) -> dict:
    """Create Milvus JSON indexing configuration for testing."""
    return {
        **json_indexing_config,
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_json_indexing",
        },
    }


@pytest.fixture
def pinecone_json_config(json_indexing_config: dict) -> dict:
    """Create Pinecone JSON indexing configuration for testing."""
    return {
        **json_indexing_config,
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-json-index",
            "namespace": "test",
            "dimension": 384,
            "metric": "cosine",
        },
    }


@pytest.fixture
def qdrant_json_config(json_indexing_config: dict) -> dict:
    """Create Qdrant JSON indexing configuration for testing."""
    return {
        **json_indexing_config,
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_json_indexing",
        },
    }


@pytest.fixture
def weaviate_json_config(json_indexing_config: dict) -> dict:
    """Create Weaviate JSON indexing configuration for testing."""
    return {
        **json_indexing_config,
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestJsonIndexing",
        },
    }


@pytest.fixture
def json_search_config_with_filters() -> dict:
    """Create JSON search configuration with filters for testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "filters": {
            "conditions": [
                {"field": "metadata.category", "value": "tech", "operator": "equals"}
            ]
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def milvus_json_search_config(json_search_config_with_filters: dict) -> dict:
    """Create Milvus JSON search configuration for testing."""
    return {
        **json_search_config_with_filters,
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_json_indexing",
        },
    }


@pytest.fixture
def pinecone_json_search_config(json_search_config_with_filters: dict) -> dict:
    """Create Pinecone JSON search configuration for testing."""
    return {
        **json_search_config_with_filters,
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-json-index",
            "namespace": "test",
        },
    }


@pytest.fixture
def qdrant_json_search_config(json_search_config_with_filters: dict) -> dict:
    """Create Qdrant JSON search configuration for testing."""
    return {
        **json_search_config_with_filters,
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_json_indexing",
        },
    }


@pytest.fixture
def weaviate_json_search_config(json_search_config_with_filters: dict) -> dict:
    """Create Weaviate JSON search configuration for testing."""
    return {
        **json_search_config_with_filters,
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestJsonIndexing",
        },
    }
