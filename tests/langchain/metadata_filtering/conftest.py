"""Fixtures for metadata filtering tests.

This module provides pytest fixtures for testing metadata filtering pipelines
across all supported vector databases. Includes sample documents with varied
metadata and database-specific configurations.

Sample data fixtures:
    sample_documents: LangChain Documents with source, id, and title metadata.

Configuration fixtures:
    milvus_metadata_filtering_config: Milvus filtering configuration.
    pinecone_metadata_filtering_config: Pinecone filtering configuration.
    qdrant_metadata_filtering_config: Qdrant filtering configuration.
    weaviate_metadata_filtering_config: Weaviate filtering configuration.

All configurations include:
    - Dataloader settings (type, limit)
    - Embeddings model configuration
    - Database connection parameters
    - RAG disabled for pure filtering tests
"""

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample LangChain documents for testing."""
    return [
        Document(
            page_content="Python is a high-level programming language",
            metadata={"source": "wiki", "id": "1", "title": "Python"},
        ),
        Document(
            page_content="Machine learning uses algorithms to learn from data",
            metadata={"source": "wiki", "id": "2", "title": "ML"},
        ),
        Document(
            page_content="Vector databases store embeddings efficiently",
            metadata={"source": "blog", "id": "3", "title": "VectorDB"},
        ),
        Document(
            page_content="LangChain is a framework for building LLM applications",
            metadata={"source": "docs", "id": "4", "title": "LangChain"},
        ),
        Document(
            page_content="Semantic search uses embeddings to find similar documents",
            metadata={"source": "blog", "id": "5", "title": "SemanticSearch"},
        ),
    ]


@pytest.fixture
def milvus_metadata_filtering_config() -> dict:
    """Create Milvus configuration for metadata filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_metadata_filtering",
            "dimension": 384,
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def pinecone_metadata_filtering_config() -> dict:
    """Create Pinecone configuration for metadata filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "pinecone": {
            "api_key": "test-api-key",
            "index_name": "test-metadata-filtering",
            "namespace": "test-namespace",
            "dimension": 384,
            "metric": "cosine",
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def qdrant_metadata_filtering_config() -> dict:
    """Create Qdrant configuration for metadata filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_metadata_filtering",
            "dimension": 384,
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def weaviate_metadata_filtering_config() -> dict:
    """Create Weaviate configuration for metadata filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestMetadataFiltering",
        },
        "rag": {"enabled": False},
    }
