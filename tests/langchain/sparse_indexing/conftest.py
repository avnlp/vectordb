"""Fixtures for sparse indexing tests.

This module provides pytest fixtures for testing sparse (BM25/Splade) indexing
pipelines across all supported vector databases. Includes mock database
connections, sparse embedder simulations, and sample documents.

Mock database fixtures:
    mock_pinecone_db: Simulated PineconeVectorDB instance.
    mock_weaviate_db: Simulated WeaviateVectorDB instance.
    mock_chroma_db: Simulated ChromaVectorDB instance.
    mock_milvus_db: Simulated MilvusVectorDB instance.
    mock_qdrant_db: Simulated QdrantVectorDB instance.

Mock embedder fixtures:
    mock_embedder: Dense embedder returning 384-dim vectors.
    mock_sparse_embedder: Sparse embedder returning indices/values dicts.

Sample data fixtures:
    sample_documents: Documents with varied content for sparse matching.

Auto-use fixtures:
    mock_sparse_embedder_init: Patches SparseEmbedder to avoid pyserini
        dependency during tests. Applied automatically to all tests.

Sparse vector format:
    {"indices": [token_ids], "values": [weights]}
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_pinecone_db() -> MagicMock:
    """Create mock PineconeVectorDB."""
    db = MagicMock()
    db.create_index.return_value = None
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_weaviate_db() -> MagicMock:
    """Create mock WeaviateVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_chroma_db() -> MagicMock:
    """Create mock ChromaVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_milvus_db() -> MagicMock:
    """Create mock MilvusVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_qdrant_db() -> MagicMock:
    """Create mock QdrantVectorDB."""
    db = MagicMock()
    db.upsert.return_value = 5
    db.query.return_value = []
    return db


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create mock embedder."""
    embedder = MagicMock()
    embedder.embed_documents.return_value = [[0.1] * 384] * 5
    embedder.embed_query.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def mock_sparse_embedder() -> MagicMock:
    """Create mock sparse embedder."""
    embedder = MagicMock()
    embedder.embed_documents.return_value = [
        {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]},
        {"indices": [1, 2, 3], "values": [0.4, 0.3, 0.3]},
        {"indices": [2, 3, 4], "values": [0.6, 0.2, 0.2]},
        {"indices": [0, 3, 4], "values": [0.3, 0.4, 0.3]},
        {"indices": [1, 4], "values": [0.5, 0.5]},
    ]
    embedder.embed_query.return_value = {
        "indices": [0, 1, 2],
        "values": [0.5, 0.3, 0.2],
    }
    return embedder


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="The quick brown fox jumps over the lazy dog",
            metadata={"source": "doc1"},
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence",
            metadata={"source": "doc2"},
        ),
        Document(
            page_content="Python is a popular programming language",
            metadata={"source": "doc3"},
        ),
        Document(
            page_content="Vector databases are designed for similarity search",
            metadata={"source": "doc4"},
        ),
        Document(
            page_content="Embeddings capture semantic meaning of text",
            metadata={"source": "doc5"},
        ),
    ]


@pytest.fixture(autouse=True)
def mock_sparse_embedder_init():
    """Auto-patch SparseEmbedder initialization to avoid pyserini import."""
    with (
        patch(
            "vectordb.langchain.utils.sparse_embeddings.SparseEmbedder.__init__",
            return_value=None,
        ),
        patch(
            "vectordb.langchain.utils.sparse_embeddings.SparseEmbedder.embed_documents"
        ) as mock_embed_docs,
        patch(
            "vectordb.langchain.utils.sparse_embeddings.SparseEmbedder.embed_query"
        ) as mock_embed_query,
    ):
        # Set default return values
        mock_embed_docs.return_value = [
            {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]},
            {"indices": [1, 2, 3], "values": [0.4, 0.3, 0.3]},
            {"indices": [2, 3, 4], "values": [0.6, 0.2, 0.2]},
            {"indices": [0, 3, 4], "values": [0.3, 0.4, 0.3]},
            {"indices": [1, 4], "values": [0.5, 0.5]},
        ]
        mock_embed_query.return_value = {
            "indices": [0, 1, 2],
            "values": [0.5, 0.3, 0.2],
        }
        yield
