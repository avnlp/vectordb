"""Shared fixtures for MMR (Maximal Marginal Relevance) tests.

This module provides pytest fixtures for testing MMR-based retrieval pipelines
across all supported vector databases. Includes sample documents, embeddings,
and database-specific configurations with MMR parameters.

Sample data fixtures:
    sample_documents: LangChain Document objects for MMR testing.
    sample_embeddings: Random 384-dim vectors (seed 42 for reproducibility).
    sample_query_embedding: Query vector for similarity computation.
    sample_mmr_candidates: Pre-scored candidate documents with embeddings.
    mmr_threshold: Default lambda threshold value (0.5).

Mock fixtures:
    mock_embedder: Patched embedder for document/query encoding.

Configuration fixtures:
    base_config: Common settings with MMR threshold and k parameters.
    milvus_config: Milvus MMR configuration.
    pinecone_config: Pinecone MMR configuration.
    qdrant_config: Qdrant MMR configuration.
    weaviate_config: Weaviate MMR configuration.
"""

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document as HaystackDocument
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
def sample_query_embedding() -> list[float]:
    """Create sample query embedding for testing."""
    import numpy as np

    np.random.seed(42)
    return np.random.randn(384).tolist()


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    with patch("vectordb.langchain.utils.EmbedderHelper.create_embedder") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = (
            ["doc1", "doc2", "doc3"],
            [[0.1] * 384 for _ in range(3)],
        )
        mock_instance.embed_query.return_value = [0.1] * 384
        mock_cls.return_value = mock_instance
        yield mock_instance


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
        "mmr": {"threshold": 0.5, "k": 5},
    }


@pytest.fixture
def milvus_config(base_config: dict) -> dict:
    """Create Milvus configuration for testing."""
    return {
        **base_config,
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_mmr",
            "dimension": 384,
        },
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
def qdrant_config(base_config: dict) -> dict:
    """Create Qdrant configuration for testing."""
    return {
        **base_config,
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "",
            "collection_name": "test_mmr",
            "dimension": 384,
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
            "collection_name": "TestMMR",
            "dimension": 384,
        },
    }


@pytest.fixture
def sample_mmr_candidates() -> list:
    """Create sample MMR candidate documents for testing."""
    candidates = []
    for i, (text, score, embedding) in enumerate(
        [
            ("Python is a high-level programming language", 0.9, [0.1] * 384),
            ("Machine learning uses algorithms to learn from data", 0.85, [0.2] * 384),
            ("Vector databases store embeddings efficiently", 0.8, [0.3] * 384),
            (
                "LangChain is a framework for building LLM applications",
                0.75,
                [0.4] * 384,
            ),
            (
                "Semantic search uses embeddings to find similar documents",
                0.7,
                [0.5] * 384,
            ),
        ]
    ):
        doc = HaystackDocument(
            id=str(i + 1),
            content=text,
            score=score,
            embedding=embedding,
            meta={},
        )
        candidates.append(doc)
    return candidates


@pytest.fixture
def mmr_threshold() -> float:
    """MMR threshold for testing."""
    return 0.5
