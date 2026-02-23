"""Shared fixtures for LangChain namespace tests.

This module provides pytest fixtures for testing namespace-based vector database
operations across all supported backends. Includes namespace context simulation,
mock database connections, and namespace-aware configurations.

Namespace context fixtures:
    sample_namespace_id: Default test namespace identifier.
    sample_namespace: Backward-compatible alias for sample_namespace_id.
    sample_namespace_config: Base namespace settings.

Mock database fixtures:
    mock_chroma_db: ChromaVectorDB with collection namespace methods.
    mock_milvus_db: MilvusVectorDB with partition namespace methods.
    mock_pinecone_db: PineconeVectorDB with native namespace methods.
    mock_qdrant_db: QdrantVectorDB with payload filter namespace methods.
    mock_weaviate_db: WeaviateVectorDB with tenant namespace methods.

Configuration fixtures:
    chroma_namespace_config: Chroma namespace configuration.
    milvus_namespace_config: Milvus namespace configuration.
    pinecone_namespace_config: Pinecone namespace configuration.
    qdrant_namespace_config: Qdrant namespace configuration.
    weaviate_namespace_config: Weaviate namespace configuration.

Sample data fixtures:
    sample_documents: Documents with namespace metadata.
    sample_embeddings: 384-dim vectors for namespace documents.
    mock_embedder: Mock embedder for query/document embeddings.
    mock_llm: Mock LLM for RAG generation.
"""

from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_namespace_id() -> str:
    """Sample namespace ID for testing."""
    return "ns_test_001"


@pytest.fixture
def sample_namespace(sample_namespace_id: str) -> str:
    """Backward-compatible alias for sample_namespace_id."""
    return sample_namespace_id


@pytest.fixture
def sample_namespace_config() -> dict:
    """Sample namespace configuration for testing."""
    return {
        "dataloader": {"type": "arc", "split": "test", "limit": 10},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "search": {"top_k": 5},
        "namespaces": {
            "enabled": True,
            "isolation_policy": "shared_instance",
        },
    }


@pytest.fixture
def pinecone_namespace_config(sample_namespace_config: dict) -> dict:
    """Create Pinecone namespace configuration for testing."""
    config = deepcopy(sample_namespace_config)
    config["pinecone"] = {
        "api_key": "test-api-key",
        "index_name": "test-index",
        "dimension": 384,
        "metric": "cosine",
    }
    return config


@pytest.fixture
def weaviate_namespace_config(sample_namespace_config: dict) -> dict:
    """Create Weaviate namespace configuration for testing."""
    config = deepcopy(sample_namespace_config)
    config["weaviate"] = {
        "url": "http://localhost:8080",
        "api_key": "",
        "collection_prefix": "ns_",
    }
    return config


@pytest.fixture
def chroma_namespace_config(sample_namespace_config: dict) -> dict:
    """Create Chroma namespace configuration for testing."""
    config = deepcopy(sample_namespace_config)
    config["chroma"] = {
        "path": "./test_chroma_data",
        "collection_prefix": "ns_",
    }
    return config


@pytest.fixture
def milvus_namespace_config(sample_namespace_config: dict) -> dict:
    """Create Milvus namespace configuration for testing."""
    config = deepcopy(sample_namespace_config)
    config["milvus"] = {
        "host": "localhost",
        "port": 19530,
        "collection_name": "namespaces",
        "dimension": 384,
    }
    return config


@pytest.fixture
def qdrant_namespace_config(sample_namespace_config: dict) -> dict:
    """Create Qdrant namespace configuration for testing."""
    config = deepcopy(sample_namespace_config)
    config["qdrant"] = {
        "url": "http://localhost:6333",
        "api_key": "",
        "collection_prefix": "ns_",
    }
    return config


@pytest.fixture
def mock_pinecone_db() -> MagicMock:
    """Create a mock PineconeVectorDB."""
    mock = MagicMock()
    mock.upsert.return_value = 3
    mock.query.return_value = ["doc1", "doc2"]
    mock.list_namespaces.return_value = ["ns_test_001", "ns_test_002"]
    mock.describe_index_stats.return_value = {
        "namespaces": {"ns_test_001": {"vector_count": 3}}
    }
    mock.delete.return_value = None
    mock.create_index.return_value = None
    return mock


@pytest.fixture
def mock_weaviate_db() -> MagicMock:
    """Create a mock WeaviateVectorDB."""
    mock = MagicMock()
    mock.upsert.return_value = 3
    mock.query.return_value = ["doc1", "doc2"]
    mock.create_tenant.return_value = None
    mock.delete_tenant.return_value = None
    mock.list_tenants.return_value = ["ns_test_001", "ns_test_002"]
    mock.with_tenant.return_value = None
    mock.collection.aggregate.over_all.return_value = MagicMock(total_count=3)
    return mock


@pytest.fixture
def mock_chroma_db() -> MagicMock:
    """Create a mock ChromaVectorDB."""
    mock = MagicMock()
    mock.upsert.return_value = 3
    mock.query.return_value = ["doc1", "doc2"]
    mock.create_collection.return_value = None
    mock.delete_collection.return_value = None
    mock.list_collections.return_value = ["ns_ns_test_001", "ns_ns_test_002"]
    mock_collection = MagicMock()
    mock_collection.count.return_value = 3
    mock._get_collection.return_value = mock_collection
    return mock


@pytest.fixture
def mock_milvus_db() -> MagicMock:
    """Create a mock MilvusVectorDB."""
    mock = MagicMock()
    mock.upsert.return_value = 3
    mock.query.return_value = ["doc1", "doc2"]
    mock.delete.return_value = None
    mock.collection_name = "namespaces"
    mock._escape_expr_string.side_effect = lambda value: value
    mock.client.query.return_value = [{"namespace": "ns_test_001"}]
    return mock


@pytest.fixture
def mock_qdrant_db() -> MagicMock:
    """Create a mock QdrantVectorDB."""
    mock = MagicMock()
    mock.upsert.return_value = 3
    mock.query.return_value = ["doc1", "doc2"]
    mock.delete.return_value = None
    mock.collection_name = "namespaces"

    record = MagicMock()
    record.payload = {"namespace": "ns_test_001"}
    mock.client.scroll.return_value = ([record], None)
    mock.client.count.return_value = MagicMock(count=3)
    return mock


@pytest.fixture
def sample_documents(sample_namespace_id: str) -> list[Document]:
    """Create sample LangChain documents for namespace testing."""
    return [
        Document(
            page_content="Python is a high-level programming language",
            metadata={"source": "wiki", "id": "1", "namespace": sample_namespace_id},
        ),
        Document(
            page_content="Machine learning uses algorithms to learn from data",
            metadata={"source": "wiki", "id": "2", "namespace": sample_namespace_id},
        ),
        Document(
            page_content="Vector databases store embeddings efficiently",
            metadata={"source": "blog", "id": "3", "namespace": sample_namespace_id},
        ),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return [np.random.randn(384).tolist() for _ in range(3)]


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Create a mock embedder."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 384
    mock.embed_documents.return_value = (
        ["doc1", "doc2", "doc3"],
        [[0.1] * 384, [0.2] * 384, [0.3] * 384],
    )
    return mock


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for RAG."""
    mock = MagicMock()
    mock.generate.return_value = "Generated answer"
    return mock
