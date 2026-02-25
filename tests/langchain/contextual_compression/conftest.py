"""Fixtures for contextual compression tests.

This module provides pytest fixtures for testing contextual compression
pipelines across all supported vector databases. Fixtures include mock
database connections, embedders, and database-specific configurations.

Mock fixtures:
    mock_chroma_db: Simulated ChromaVectorDB instance.
    mock_milvus_db: Simulated MilvusVectorDB instance.
    mock_pinecone_db: Simulated PineconeVectorDB instance.
    mock_qdrant_db: Simulated QdrantVectorDB instance.
    mock_weaviate_db: Simulated WeaviateVectorDB instance.
    mock_embedder: Mock embedder returning 384-dimensional vectors.

Configuration fixtures:
    contextual_compression_config: Base Chroma configuration.
    milvus_contextual_compression_config: Milvus-specific settings.
    pinecone_contextual_compression_config: Pinecone-specific settings.
    qdrant_contextual_compression_config: Qdrant-specific settings.
    weaviate_contextual_compression_config: Weaviate-specific settings.

Sample data:
    sample_documents: LangChain Document objects with varied metadata.
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
def contextual_compression_config() -> dict:
    """Create base configuration for contextual compression testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "chroma": {
            "path": "./test_chroma_data",
            "collection_name": "test_contextual_compression",
        },
    }


@pytest.fixture
def milvus_contextual_compression_config() -> dict:
    """Create Milvus configuration for contextual compression testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_contextual_compression",
            "dimension": 384,
        },
    }


@pytest.fixture
def pinecone_contextual_compression_config() -> dict:
    """Create Pinecone configuration for contextual compression testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "pinecone": {
            "api_key": "test-api-key",
            "index_name": "test-contextual-compression",
            "namespace": "test-namespace",
            "dimension": 384,
            "metric": "cosine",
        },
    }


@pytest.fixture
def qdrant_contextual_compression_config() -> dict:
    """Create Qdrant configuration for contextual compression testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_contextual_compression",
            "dimension": 384,
        },
    }


@pytest.fixture
def weaviate_contextual_compression_config() -> dict:
    """Create Weaviate configuration for contextual compression testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestContextualCompression",
        },
    }
