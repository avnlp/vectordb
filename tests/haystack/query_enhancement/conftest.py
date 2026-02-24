"""Shared fixtures for query enhancement pipeline tests.

This module provides pytest fixtures for testing query enhancement
implementations in Haystack. Query enhancement improves retrieval by
transforming queries using techniques like multi-query generation,
HyDE (Hypothetical Document Embeddings), and query decomposition.

Fixtures:
    mock_document: Mocked Haystack Document with content and embedding.
    sample_config: Configuration dictionary with query enhancement settings
        including multi-query, HyDE parameters, and LLM configuration.
    provider_fixtures: Parametrized provider configuration for all databases.

Note:
    Query enhancement tests validate that transformed queries improve
    retrieval recall by generating semantically diverse query variants
    or hypothetical answer documents for embedding-based search.
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest
from haystack import Document


@pytest.fixture
def mock_document() -> Mock:
    """Create a mock Haystack Document for testing.

    Returns:
        Mock Document object with content, metadata, and embedding
        attributes for testing query enhancement pipeline outputs.
    """
    doc = Mock()
    doc.content = "test content"
    doc.meta = {}
    doc.embedding = [0.1, 0.2, 0.3]
    return doc


@pytest.fixture
def sample_config() -> dict:
    """Create sample configuration for query enhancement tests.

    Returns:
        Configuration dictionary with dataloader, embeddings, query
        enhancement, database, and RAG settings for pipeline testing.
    """
    return {
        "dataloader": {"type": "test", "params": {"limit": 10}},
        "embeddings": {"model": "all-MiniLM-L6-v2", "params": {}},
        "query_enhancement": {
            "type": "multi_query",
            "num_queries": 3,
            "num_hyde_docs": 3,
            "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 10,
        },
        "pinecone": {
            "api_key": "test-api-key",
            "index_name": "test-index",
            "namespace": "test-namespace",
        },
        "logging": {"level": "INFO", "name": "test"},
        "rag": {"enabled": False},
    }


@dataclass
class ProviderFixture:
    """Container for provider-specific test fixtures."""

    name: str
    config: dict[str, Any]
    config_with_rag: dict[str, Any]
    indexing_pipeline_path: str
    search_pipeline_path: str
    indexing_db_mock_path: str
    search_db_mock_path: str
    integration_env_var: str


def _create_base_config() -> dict[str, Any]:
    """Create base configuration shared across all providers."""
    return {
        "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "dataloader": {"name": "triviaqa", "limit": 10},
        "query_enhancement": {
            "type": "multi_query",
            "num_queries": 3,
            "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
        },
        "rag": {"enabled": False},
    }


def _get_pinecone_fixture() -> ProviderFixture:
    config = _create_base_config()
    config["pinecone"] = {
        "api_key": "test-key",
        "index_name": "test-index",
        "namespace": "test-namespace",
    }
    config_with_rag = config.copy()
    config_with_rag["rag"] = {
        "enabled": True,
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    }
    return ProviderFixture(
        name="pinecone",
        config=config,
        config_with_rag=config_with_rag,
        indexing_pipeline_path="vectordb.haystack.query_enhancement.indexing.pinecone.PineconeQueryEnhancementIndexingPipeline",
        search_pipeline_path="vectordb.haystack.query_enhancement.search.pinecone.PineconeQueryEnhancementSearchPipeline",
        indexing_db_mock_path="vectordb.haystack.query_enhancement.indexing.pinecone.PineconeVectorDB",
        search_db_mock_path="vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB",
        integration_env_var="PINECONE_API_KEY",
    )


def _get_chroma_fixture() -> ProviderFixture:
    config = _create_base_config()
    config["rag"] = {
        "enabled": True,
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    }
    config["chroma"] = {
        "persist_directory": "/tmp/chroma_test",
        "collection_name": "test_collection",
    }
    config_with_rag = config.copy()
    return ProviderFixture(
        name="chroma",
        config=config,
        config_with_rag=config_with_rag,
        indexing_pipeline_path="vectordb.haystack.query_enhancement.indexing.chroma.ChromaQueryEnhancementIndexingPipeline",
        search_pipeline_path="vectordb.haystack.query_enhancement.search.chroma.ChromaQueryEnhancementSearchPipeline",
        indexing_db_mock_path="vectordb.haystack.query_enhancement.indexing.chroma.ChromaVectorDB",
        search_db_mock_path="vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB",
        integration_env_var="CHROMA_PERSIST_DIR",
    )


def _get_qdrant_fixture() -> ProviderFixture:
    config = _create_base_config()
    config["rag"] = {
        "enabled": True,
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    }
    config["qdrant"] = {
        "url": "http://localhost:6333",
        "collection_name": "test_collection",
    }
    config_with_rag = config.copy()
    return ProviderFixture(
        name="qdrant",
        config=config,
        config_with_rag=config_with_rag,
        indexing_pipeline_path="vectordb.haystack.query_enhancement.indexing.qdrant.QdrantQueryEnhancementIndexingPipeline",
        search_pipeline_path="vectordb.haystack.query_enhancement.search.qdrant.QdrantQueryEnhancementSearchPipeline",
        indexing_db_mock_path="vectordb.haystack.query_enhancement.indexing.qdrant.QdrantVectorDB",
        search_db_mock_path="vectordb.haystack.query_enhancement.search.qdrant.QdrantVectorDB",
        integration_env_var="QDRANT_URL",
    )


def _get_weaviate_fixture() -> ProviderFixture:
    config = _create_base_config()
    config["weaviate"] = {
        "url": "http://localhost:8080",
        "api_key": "test-key",
        "class_name": "TestQueryEnhancement",
    }
    config_with_rag = config.copy()
    config_with_rag["rag"] = {
        "enabled": True,
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    }
    return ProviderFixture(
        name="weaviate",
        config=config,
        config_with_rag=config_with_rag,
        indexing_pipeline_path="vectordb.haystack.query_enhancement.indexing.weaviate.WeaviateQueryEnhancementIndexingPipeline",
        search_pipeline_path="vectordb.haystack.query_enhancement.search.weaviate.WeaviateQueryEnhancementSearchPipeline",
        indexing_db_mock_path="vectordb.haystack.query_enhancement.indexing.weaviate.WeaviateVectorDB",
        search_db_mock_path="vectordb.haystack.query_enhancement.search.weaviate.WeaviateVectorDB",
        integration_env_var="WEAVIATE_URL",
    )


def _get_milvus_fixture() -> ProviderFixture:
    config = _create_base_config()
    config["rag"] = {
        "enabled": True,
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    }
    config["milvus"] = {
        "uri": "http://localhost:19530",
        "collection_name": "test_collection",
    }
    config_with_rag = config.copy()
    return ProviderFixture(
        name="milvus",
        config=config,
        config_with_rag=config_with_rag,
        indexing_pipeline_path="vectordb.haystack.query_enhancement.indexing.milvus.MilvusQueryEnhancementIndexingPipeline",
        search_pipeline_path="vectordb.haystack.query_enhancement.search.milvus.MilvusQueryEnhancementSearchPipeline",
        indexing_db_mock_path="vectordb.haystack.query_enhancement.indexing.milvus.MilvusVectorDB",
        search_db_mock_path="vectordb.haystack.query_enhancement.search.milvus.MilvusVectorDB",
        integration_env_var="MILVUS_URI",
    )


PROVIDER_FIXTURES = {
    "pinecone": _get_pinecone_fixture,
    "chroma": _get_chroma_fixture,
    "qdrant": _get_qdrant_fixture,
    "weaviate": _get_weaviate_fixture,
    "milvus": _get_milvus_fixture,
}


@pytest.fixture
def mock_documents() -> list[Document]:
    """Create mock documents with embeddings.

    Returns:
        List of Document objects with content and embedding attributes.
    """
    return [
        Document(content=f"Test document {i}", embedding=[0.1] * 384) for i in range(5)
    ]


@pytest.fixture(params=list(PROVIDER_FIXTURES.keys()))
def provider_fixture(request: pytest.FixtureRequest) -> ProviderFixture:
    """Parametrized fixture for all database providers.

    Returns:
        ProviderFixture containing provider-specific configuration and paths.
    """
    return PROVIDER_FIXTURES[request.param]()


@pytest.fixture
def pinecone_fixture() -> ProviderFixture:
    """Pinecone-specific fixture for provider-specific tests."""
    return _get_pinecone_fixture()


@pytest.fixture
def chroma_fixture() -> ProviderFixture:
    """Chroma-specific fixture for provider-specific tests."""
    return _get_chroma_fixture()


@pytest.fixture
def qdrant_fixture() -> ProviderFixture:
    """Qdrant-specific fixture for provider-specific tests."""
    return _get_qdrant_fixture()


@pytest.fixture
def weaviate_fixture() -> ProviderFixture:
    """Weaviate-specific fixture for provider-specific tests."""
    return _get_weaviate_fixture()


@pytest.fixture
def milvus_fixture() -> ProviderFixture:
    """Milvus-specific fixture for provider-specific tests."""
    return _get_milvus_fixture()
