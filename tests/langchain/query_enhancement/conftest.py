"""Fixtures for query enhancement tests.

This module provides pytest fixtures for testing query enhancement pipelines
across all supported vector databases. Includes mock database connections,
sample documents, and database-specific configurations.

Mock database fixtures:
    mock_chroma_db: Simulated ChromaVectorDB instance.
    mock_milvus_db: Simulated MilvusVectorDB instance.
    mock_pinecone_db: Simulated PineconeVectorDB instance.
    mock_qdrant_db: Simulated QdrantVectorDB instance.
    mock_weaviate_db: Simulated WeaviateVectorDB instance.
    mock_embedder: Mock embedder returning 384-dimensional vectors.

Sample data fixtures:
    sample_documents: LangChain Documents for query enhancement testing.

Configuration fixtures:
    chroma_query_enhancement_config: Chroma configuration (RAG disabled).
    milvus_query_enhancement_config: Milvus configuration.
    pinecone_query_enhancement_config: Pinecone configuration.
    qdrant_query_enhancement_config: Qdrant configuration.
    weaviate_query_enhancement_config: Weaviate configuration.

Note:
    All configurations have RAG disabled to isolate query enhancement
    logic from answer generation during testing.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_chroma_db() -> MagicMock:
    """Create mock ChromaVectorDB."""
    db = MagicMock()
    db.create_collection.return_value = None
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_milvus_db() -> MagicMock:
    """Create mock MilvusVectorDB."""
    db = MagicMock()
    db.create_collection.return_value = None
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_pinecone_db() -> MagicMock:
    """Create mock PineconeVectorDB."""
    db = MagicMock()
    db.create_index.return_value = True
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_qdrant_db() -> MagicMock:
    """Create mock QdrantVectorDB."""
    db = MagicMock()
    db.create_collection.return_value = None
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_weaviate_db() -> MagicMock:
    """Create mock WeaviateVectorDB."""
    db = MagicMock()
    db.create_collection.return_value = None
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create mock embedder."""
    embedder = MagicMock()
    embedder.embed_documents.return_value = ([[0.1] * 384] * 5, [[0.1] * 384] * 5)
    embedder.embed_query.return_value = [0.1] * 384
    return embedder


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
def chroma_query_enhancement_config() -> dict:
    """Create Chroma configuration for query enhancement testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "chroma": {
            "path": "./test_chroma_data",
            "collection_name": "test_query_enhancement",
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def milvus_query_enhancement_config() -> dict:
    """Create Milvus configuration for query enhancement testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_query_enhancement",
            "dimension": 384,
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def pinecone_query_enhancement_config() -> dict:
    """Create Pinecone configuration for query enhancement testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "pinecone": {
            "api_key": "test-api-key",
            "index_name": "test-query-enhancement",
            "namespace": "test-namespace",
            "dimension": 384,
            "metric": "cosine",
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def qdrant_query_enhancement_config() -> dict:
    """Create Qdrant configuration for query enhancement testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_query_enhancement",
            "dimension": 384,
        },
        "rag": {"enabled": False},
    }


@pytest.fixture
def weaviate_query_enhancement_config() -> dict:
    """Create Weaviate configuration for query enhancement testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestQueryEnhancement",
        },
        "rag": {"enabled": False},
    }
