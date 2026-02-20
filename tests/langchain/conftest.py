"""Shared fixtures for LangChain tests.

This module provides pytest fixtures designed for testing LangChain
integrations with vector databases. These fixtures simulate LangChain
components and provide test configuration templates.

Fixtures:
    sample_documents: LangChain Document objects for testing retrieval.
    sample_embeddings: Sample 384-dimensional embedding vectors (numpy-based).
    base_config: Base configuration template for tests.
    pinecone_config: Pinecone-specific configuration.
    weaviate_config: Weaviate-specific configuration.
    chroma_config: Chroma-specific configuration.
    milvus_config: Milvus-specific configuration.
    qdrant_config: Qdrant-specific configuration.
    agentic_rag_config: Agentic RAG pipeline configuration.
    cost_optimized_rag_config: Cost-optimized RAG configuration.

Note:
    Sample embeddings use numpy with seed 42 for reproducibility.
    All embeddings are 384-dimensional to match the default
    sentence-transformers/all-MiniLM-L6-v2 model.
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
def sample_embeddings() -> list[list[float]]:
    """Create sample embeddings for testing."""
    import numpy as np

    np.random.seed(42)
    return [np.random.randn(384).tolist() for _ in range(5)]


@pytest.fixture
def base_config() -> dict:
    """Create base configuration for testing."""
    return {
        "dataloader": {"type": "arc", "split": "test", "limit": 10},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "search": {"top_k": 5},
    }


@pytest.fixture
def pinecone_config(base_config: dict) -> dict:
    """Create Pinecone configuration for testing."""
    return {
        **base_config,
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-index",
            "namespace": "test",
            "dimension": 384,
            "metric": "cosine",
        },
    }


@pytest.fixture
def weaviate_config(base_config: dict) -> dict:
    """Create Weaviate configuration for testing."""
    return {
        **base_config,
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestSemanticSearch",
        },
    }


@pytest.fixture
def chroma_config(base_config: dict) -> dict:
    """Create Chroma configuration for testing."""
    return {
        **base_config,
        "chroma": {
            "path": "./test_chroma_data",
            "collection_name": "test_semantic_search",
        },
    }


@pytest.fixture
def milvus_config(base_config: dict) -> dict:
    """Create Milvus configuration for testing."""
    return {
        **base_config,
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_semantic_search",
            "dimension": 384,
        },
    }


@pytest.fixture
def qdrant_config(base_config: dict) -> dict:
    """Create Qdrant configuration for testing."""
    return {
        **base_config,
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_semantic_search",
        },
    }


@pytest.fixture
def agentic_rag_config() -> dict:
    """Create agentic RAG configuration for testing."""
    return {
        "dataloader": {"type": "arc", "split": "test", "limit": 10},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "search": {"top_k": 5, "score_threshold": 0.5},
        "chroma": {
            "path": "./test_chroma_data",
            "collection_name": "test_agentic_rag",
        },
        "chunking": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
    }


@pytest.fixture
def cost_optimized_rag_config() -> dict:
    """Create cost-optimized RAG configuration for testing."""
    return {
        "dataloader": {"type": "arc", "split": "test", "limit": 10},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "search": {"top_k": 5, "score_threshold": 0.5},
        "chroma": {
            "path": "./test_chroma_data",
            "collection_name": "test_cost_optimized_rag",
        },
        "sparse_embeddings": {
            "model": "prithivida/Splade_PP_en_v1",
        },
        "chunking": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
    }
