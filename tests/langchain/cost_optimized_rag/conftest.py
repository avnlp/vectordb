"""Shared fixtures for cost-optimized RAG tests.

This module provides pytest fixtures for testing cost-optimized RAG pipelines
across all supported vector databases. Includes mock embedders, sample data,
and database-specific configuration templates.

Sample data fixtures:
    sample_documents: LangChain Document objects for testing.
    sample_embeddings: Random 384-dim vectors (seed 42 for reproducibility).
    sample_dense_embeddings: Uniform 384-dim dense vectors.
    sample_sparse_embeddings: Sparse vector dicts with indices/values.

Mock fixtures:
    mock_dense_embedder: Patched DenseEmbedder returning fixed vectors.
    mock_sparse_embedder: Patched SparseEmbedder for hybrid search tests.

Configuration fixtures:
    base_config: Common dataloader/embeddings/search settings.
    milvus_config: Milvus connection and collection settings.
    pinecone_config: Pinecone API and index configuration.
    qdrant_config: Qdrant connection and collection settings.
    weaviate_config: Weaviate connection with alpha fusion parameter.
"""

from unittest.mock import MagicMock, patch

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
def sample_dense_embeddings() -> list[list[float]]:
    """Create sample dense embeddings for testing."""
    import numpy as np

    np.random.seed(42)
    return [[0.1] * 384 for _ in range(5)]


@pytest.fixture
def sample_sparse_embeddings() -> list[dict]:
    """Create sample sparse embeddings for testing."""
    return [{"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]} for _ in range(5)]


@pytest.fixture
def mock_dense_embedder():
    """Create a mock dense embedder."""
    with patch("vectordb.langchain.utils.embeddings.DenseEmbedder") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.embed.return_value = [[0.1] * 384 for _ in range(5)]
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_sparse_embedder():
    """Create a mock sparse embedder."""
    with patch("vectordb.langchain.utils.embeddings.SparseEmbedder") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.embed_documents.return_value = [
            {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]} for _ in range(5)
        ]
        mock_instance.embed_query.return_value = {
            "indices": [0, 1, 2],
            "values": [0.5, 0.3, 0.2],
        }
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
        "search": {"top_k": 5, "rrf_k": 60},
    }


@pytest.fixture
def milvus_config(base_config: dict) -> dict:
    """Create Milvus configuration for testing."""
    return {
        **base_config,
        "milvus": {
            "uri": "http://localhost:19530",
            "db_name": "default",
            "collection_name": "test_cost_optimized_rag",
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
            "collection_name": "test_cost_optimized_rag",
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
            "collection_name": "TestCostOptimizedRAG",
            "dimension": 384,
        },
        "search": {"top_k": 5, "rrf_k": 60, "alpha": 0.5},
    }
