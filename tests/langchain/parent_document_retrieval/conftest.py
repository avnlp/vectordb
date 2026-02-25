"""Fixtures for parent document retrieval tests.

This module provides pytest fixtures for testing parent document retrieval
pipelines across all supported vector databases. Includes mock database
connections, sample documents, and chunking configurations.

Mock database fixtures:
    mock_milvus_db: Simulated MilvusVectorDB instance.
    mock_pinecone_db: Simulated PineconeVectorDB instance.
    mock_qdrant_db: Simulated QdrantVectorDB instance.
    mock_weaviate_db: Simulated WeaviateVectorDB instance.
    mock_embedder: Mock embedder returning 384-dimensional vectors.

Sample data fixtures:
    sample_documents: LangChain Documents representing parent documents.

Configuration fixtures:
    milvus_parent_document_retrieval_config: Milvus with chunking settings.
    pinecone_parent_document_retrieval_config: Pinecone configuration.
    qdrant_parent_document_retrieval_config: Qdrant configuration.
    weaviate_parent_document_retrieval_config: Weaviate configuration.

Chunking parameters (100 chars, 20 overlap):
    chunk_size: Maximum characters per child chunk.
    chunk_overlap: Character overlap between adjacent chunks.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


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
            page_content="Python is a high-level programming language with simple syntax.",
            metadata={"source": "wiki", "id": "1", "title": "Python"},
        ),
        Document(
            page_content="Machine learning uses algorithms to learn from data patterns.",
            metadata={"source": "wiki", "id": "2", "title": "ML"},
        ),
        Document(
            page_content="Vector databases store embeddings efficiently for search.",
            metadata={"source": "blog", "id": "3", "title": "VectorDB"},
        ),
        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "docs", "id": "4", "title": "LangChain"},
        ),
        Document(
            page_content="Semantic search uses embeddings to find similar documents.",
            metadata={"source": "blog", "id": "5", "title": "SemanticSearch"},
        ),
    ]


@pytest.fixture
def milvus_parent_document_retrieval_config() -> dict:
    """Create Milvus configuration for parent document retrieval testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "milvus": {
            "uri": "http://localhost:19530",
            "db_name": "default",
            "collection_name": "test_parent_document_retrieval",
            "dimension": 384,
        },
        "chunking": {
            "chunk_size": 100,
            "chunk_overlap": 20,
        },
    }


@pytest.fixture
def pinecone_parent_document_retrieval_config() -> dict:
    """Create Pinecone configuration for parent document retrieval testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "pinecone": {
            "api_key": "test-api-key",
            "index_name": "test-parent-document-retrieval",
            "namespace": "test-namespace",
            "dimension": 384,
            "metric": "cosine",
        },
        "chunking": {
            "chunk_size": 100,
            "chunk_overlap": 20,
        },
    }


@pytest.fixture
def qdrant_parent_document_retrieval_config() -> dict:
    """Create Qdrant configuration for parent document retrieval testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_parent_document_retrieval",
            "dimension": 384,
        },
        "chunking": {
            "chunk_size": 100,
            "chunk_overlap": 20,
        },
    }


@pytest.fixture
def weaviate_parent_document_retrieval_config() -> dict:
    """Create Weaviate configuration for parent document retrieval testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "",
            "collection_name": "TestParentDocumentRetrieval",
        },
        "chunking": {
            "chunk_size": 100,
            "chunk_overlap": 20,
        },
    }
