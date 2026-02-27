"""Fixtures for diversity filtering tests.

This module provides pytest fixtures for testing diversity filtering pipelines
across all supported vector databases. Includes sample documents and
database-specific configurations with diversity parameters.

Sample data fixtures:
    sample_documents: LangChain Document objects with varied sources.

Configuration fixtures:
    milvus_diversity_filtering_config: Milvus with threshold-based filtering.
    pinecone_diversity_filtering_config: Pinecone diversity configuration.
    qdrant_diversity_filtering_config: Qdrant diversity configuration.
    weaviate_diversity_filtering_config: Weaviate diversity configuration.

Diversity parameters:
    method: Filtering algorithm (threshold, mmr, clustering).
    max_documents: Maximum documents to return after filtering.
    similarity_threshold: Minimum inter-document distance threshold.
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
def milvus_diversity_filtering_config() -> dict:
    """Create Milvus configuration for diversity filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_diversity_filtering",
            "dimension": 384,
        },
        "diversity": {
            "method": "threshold",
            "max_documents": 5,
            "similarity_threshold": 0.7,
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def pinecone_diversity_filtering_config() -> dict:
    """Create Pinecone configuration for diversity filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "pinecone": {
            "api_key": "test-api-key",
            "index_name": "test-diversity-filtering",
            "namespace": "test-namespace",
            "dimension": 384,
            "metric": "cosine",
        },
        "diversity": {
            "method": "threshold",
            "max_documents": 5,
            "similarity_threshold": 0.7,
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def qdrant_diversity_filtering_config() -> dict:
    """Create Qdrant configuration for diversity filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_diversity_filtering",
            "dimension": 384,
        },
        "diversity": {
            "method": "threshold",
            "max_documents": 5,
            "similarity_threshold": 0.7,
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def weaviate_diversity_filtering_config() -> dict:
    """Create Weaviate configuration for diversity filtering testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestDiversityFiltering",
        },
        "diversity": {
            "method": "threshold",
            "max_documents": 5,
            "similarity_threshold": 0.7,
        },
        "rag": {"enabled": False},
    }
