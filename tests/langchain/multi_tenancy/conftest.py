"""Shared fixtures for LangChain multi-tenancy tests.

This module provides pytest fixtures for testing multi-tenant vector database
operations across all supported backends. Includes tenant context simulation,
mock database connections, and tenant-aware configurations.

Tenant context fixtures:
    mock_tenant_context: Simulated tenant with ID, user, and metadata.
    sample_tenant_id: Default test tenant identifier.
    sample_multi_tenant_config: Base multi-tenancy settings.

Mock database fixtures:
    mock_chroma_db: ChromaVectorDB with tenant collection methods.
    mock_milvus_db: MilvusVectorDB with partition-based tenancy.
    mock_pinecone_db: PineconeVectorDB with namespace tenancy.
    mock_qdrant_db: QdrantVectorDB with collection-based tenancy.
    mock_weaviate_db: WeaviateVectorDB with class-based tenancy.

Configuration fixtures:
    chroma_multi_tenant_config: Chroma tenant isolation settings.
    milvus_multi_tenant_config: Milvus partition configuration.
    pinecone_multi_tenant_config: Pinecone namespace configuration.
    qdrant_multi_tenant_config: Qdrant collection prefix settings.
    weaviate_multi_tenant_config: Weaviate collection prefix settings.

Sample data fixtures:
    sample_documents: Documents with tenant_id in metadata.
    sample_embeddings: 384-dim vectors for tenant documents.
    mock_embedder: Mock embedder for tenant-aware embedding.
    mock_llm: Mock LLM for tenant-scoped RAG generation.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_tenant_context() -> MagicMock:
    """Create a mock tenant context."""
    mock = MagicMock()
    mock.tenant_id = "tenant_123"
    mock.user_id = "user_456"
    mock.metadata = {"tier": "premium"}
    return mock


@pytest.fixture
def sample_tenant_id() -> str:
    """Sample tenant ID for testing."""
    return "tenant_test_001"


@pytest.fixture
def sample_multi_tenant_config() -> dict:
    """Sample multi-tenant configuration for testing."""
    return {
        "dataloader": {"type": "arc", "split": "test", "limit": 10},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "search": {"top_k": 5},
        "multitenancy": {
            "enabled": True,
            "tenant_field": "tenant_id",
            "isolation_policy": "shared_instance",
        },
    }


@pytest.fixture
def chroma_multi_tenant_config(sample_multi_tenant_config: dict) -> dict:
    """Create Chroma multi-tenant configuration for testing."""
    config = sample_multi_tenant_config.copy()
    config["chroma"] = {
        "path": "./test_chroma_data",
        "collection_prefix": "tenant_",
    }
    return config


@pytest.fixture
def milvus_multi_tenant_config(sample_multi_tenant_config: dict) -> dict:
    """Create Milvus multi-tenant configuration for testing."""
    config = sample_multi_tenant_config.copy()
    config["milvus"] = {
        "host": "localhost",
        "port": 19530,
        "collection_name": "multi_tenancy",
        "dimension": 384,
    }
    return config


@pytest.fixture
def pinecone_multi_tenant_config(sample_multi_tenant_config: dict) -> dict:
    """Create Pinecone multi-tenant configuration for testing."""
    config = sample_multi_tenant_config.copy()
    config["pinecone"] = {
        "api_key": "test-api-key",
        "index_name": "test-index",
        "dimension": 384,
        "metric": "cosine",
    }
    return config


@pytest.fixture
def qdrant_multi_tenant_config(sample_multi_tenant_config: dict) -> dict:
    """Create Qdrant multi-tenant configuration for testing."""
    config = sample_multi_tenant_config.copy()
    config["qdrant"] = {
        "url": "http://localhost:6333",
        "api_key": "",
        "collection_prefix": "tenant_",
    }
    return config


@pytest.fixture
def weaviate_multi_tenant_config(sample_multi_tenant_config: dict) -> dict:
    """Create Weaviate multi-tenant configuration for testing."""
    config = sample_multi_tenant_config.copy()
    config["weaviate"] = {
        "url": "http://localhost:8080",
        "api_key": "",
        "collection_prefix": "tenant_",
    }
    return config


@pytest.fixture
def mock_chroma_db() -> MagicMock:
    """Create a mock ChromaVectorDB."""
    mock = MagicMock()
    mock.query.return_value = ["doc1", "doc2"]
    mock.index_for_tenant.return_value = 5
    mock.upsert.return_value = 5
    mock.get_collections.return_value = ["tenant_tenant1", "tenant_tenant2"]
    mock.delete_collection.return_value = None
    return mock


@pytest.fixture
def mock_milvus_db() -> MagicMock:
    """Create a mock MilvusVectorDB."""
    mock = MagicMock()
    mock.query.return_value = ["doc1", "doc2"]
    mock.index_for_tenant.return_value = 5
    mock.upsert.return_value = 5
    mock.get_partitions.return_value = ["tenant1", "tenant2"]
    mock.delete_partition.return_value = None
    return mock


@pytest.fixture
def mock_pinecone_db() -> MagicMock:
    """Create a mock PineconeVectorDB."""
    mock = MagicMock()
    mock.query.return_value = ["doc1", "doc2"]
    mock.index_for_tenant.return_value = 5
    mock.upsert.return_value = 5
    mock.get_index_stats.return_value = {"namespaces": {"tenant1": {}, "tenant2": {}}}
    mock.delete_namespace.return_value = None
    mock.create_index.return_value = None
    return mock


@pytest.fixture
def mock_qdrant_db() -> MagicMock:
    """Create a mock QdrantVectorDB."""
    mock = MagicMock()
    mock.query.return_value = ["doc1", "doc2"]
    mock.index_for_tenant.return_value = 5
    mock.upsert.return_value = 5
    mock.get_collections.return_value = ["tenant_tenant1", "tenant_tenant2"]
    mock.delete_collection.return_value = None
    return mock


@pytest.fixture
def mock_weaviate_db() -> MagicMock:
    """Create a mock WeaviateVectorDB."""
    mock = MagicMock()
    mock.query.return_value = ["doc1", "doc2"]
    mock.index_for_tenant.return_value = 5
    mock.upsert.return_value = 5
    mock.get_collections.return_value = ["tenant_tenant1", "tenant_tenant2"]
    mock.delete_collection.return_value = None
    return mock


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample LangChain documents for multi-tenancy testing."""
    return [
        Document(
            page_content="Python is a high-level programming language",
            metadata={"source": "wiki", "id": "1", "tenant_id": "tenant_test_001"},
        ),
        Document(
            page_content="Machine learning uses algorithms to learn from data",
            metadata={"source": "wiki", "id": "2", "tenant_id": "tenant_test_001"},
        ),
        Document(
            page_content="Vector databases store embeddings efficiently",
            metadata={"source": "blog", "id": "3", "tenant_id": "tenant_test_001"},
        ),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample embeddings for testing."""
    import numpy as np

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
