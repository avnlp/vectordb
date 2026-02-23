"""Fixtures for agentic RAG search tests.

This module provides pytest fixtures for testing agentic RAG pipeline
components across all supported vector databases. Fixtures simulate
database connections, embedders, LLMs, and pipeline components.

Mock fixtures:
    mock_chroma_db: Simulated ChromaVectorDB with query/upsert methods.
    mock_milvus_db: Simulated MilvusVectorDB with partition support.
    mock_pinecone_db: Simulated PineconeVectorDB with namespace handling.
    mock_qdrant_db: Simulated QdrantVectorDB with collection operations.
    mock_weaviate_db: Simulated WeaviateVectorDB instance.
    mock_embedder: Mock embedder returning 384-dim vectors.
    mock_llm: Mock LLM with invoke/generate methods.
    mock_reranker: Mock reranker for document scoring.
    mock_agent_router: Mock router returning action decisions.
    mock_context_compressor: Mock compressor for document filtering.

Configuration fixtures:
    milvus_search_config: Milvus agentic RAG configuration.
    pinecone_search_config: Pinecone agentic RAG configuration.
    qdrant_search_config: Qdrant agentic RAG configuration.
    weaviate_search_config: Weaviate agentic RAG configuration.

Sample data fixtures:
    sample_documents: LangChain Document objects for testing.
    sample_query_result: Expected query result structure.
    sample_search_documents: Documents formatted for search results.
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
    db.create_index.return_value = None
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
def mock_llm() -> MagicMock:
    """Create mock LLM."""
    llm = MagicMock()
    llm.api_key = "test-api-key"
    llm.invoke.return_value = MagicMock(content="Test answer")
    llm.generate.return_value = "Generated answer"
    return llm


@pytest.fixture
def mock_reranker() -> MagicMock:
    """Create mock reranker."""
    reranker = MagicMock()
    reranker.rerank.return_value = []
    return reranker


@pytest.fixture
def mock_agent_router() -> MagicMock:
    """Create mock agent router."""
    router = MagicMock()
    router.route.return_value = {
        "action": "search",
        "reasoning": "Need to search for relevant documents",
    }
    return router


@pytest.fixture
def mock_context_compressor() -> MagicMock:
    """Create mock context compressor."""
    compressor = MagicMock()
    compressor.compress.return_value = []
    return compressor


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
def milvus_search_config() -> dict:
    """Create Milvus configuration for agentic RAG search testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_agentic_rag",
            "dimension": 384,
        },
        "llm": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "test-key",
        },
        "agentic": {
            "max_iterations": 3,
            "compression_mode": "reranking",
            "router_model": "llama-3.3-70b-versatile",
        },
        "reranker": {
            "provider": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    }


@pytest.fixture
def pinecone_search_config() -> dict:
    """Create Pinecone configuration for agentic RAG search testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "pinecone": {
            "api_key": "test-key",
            "index_name": "test-index",
            "namespace": "test",
            "dimension": 384,
        },
        "llm": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "test-key",
        },
        "agentic": {
            "max_iterations": 3,
            "compression_mode": "reranking",
            "router_model": "llama-3.3-70b-versatile",
        },
        "reranker": {
            "provider": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    }


@pytest.fixture
def qdrant_search_config() -> dict:
    """Create Qdrant configuration for agentic RAG search testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "qdrant": {
            "url": "http://localhost:6333",
            "api_key": "test-key",
            "collection_name": "test_agentic_rag",
            "dimension": 384,
        },
        "llm": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "test-key",
        },
        "agentic": {
            "max_iterations": 3,
            "compression_mode": "reranking",
            "router_model": "llama-3.3-70b-versatile",
        },
        "reranker": {
            "provider": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    }


@pytest.fixture
def weaviate_search_config() -> dict:
    """Create Weaviate configuration for agentic RAG search testing."""
    return {
        "dataloader": {"type": "arc", "limit": 10},
        "embeddings": {"model": "test-model", "device": "cpu"},
        "weaviate": {
            "url": "http://localhost:8080",
            "api_key": "test-key",
            "collection_name": "TestAgenticRAG",
        },
        "llm": {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "test-key",
        },
        "agentic": {
            "max_iterations": 3,
            "compression_mode": "reranking",
            "router_model": "llama-3.3-70b-versatile",
        },
        "reranker": {
            "provider": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
    }


@pytest.fixture
def sample_query_result() -> dict:
    """Create sample query result."""
    return {
        "query": "test query",
        "documents": [],
        "final_answer": "Test answer",
        "intermediate_steps": [],
        "reasoning": [],
    }


@pytest.fixture
def sample_search_documents(sample_documents) -> list[Document]:
    """Create sample documents for search results."""
    return sample_documents
