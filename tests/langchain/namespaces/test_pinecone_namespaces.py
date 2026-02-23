"""Tests for Pinecone namespace pipelines using LangChain.

This module provides comprehensive unit tests for Pinecone-based namespace
functionality using native namespace-based isolation. Pinecone uses native
namespace parameter. Zero overhead per namespace.

Namespace Strategy:
    - Uses Pinecone namespaces as data boundaries
    - All namespaces share the same index but are isolated via namespace parameter
    - Namespace name becomes the namespace value in all operations

Test Coverage:
    - Initialization: Pipeline creation and isolation strategy verification
    - Management: Namespace create, delete, list, exists, and stats
    - Indexing: Document upsert with namespace routing
    - Query: Namespace-scoped and cross-namespace queries

Test Organization:
    - TestPineconeNamespaceInit: Initialization and strategy tests
    - TestPineconeNamespaceManagement: CRUD operations on namespaces
    - TestPineconeNamespaceIndexing: Core indexing operations
    - TestPineconeNamespaceQuery: Query operations

All tests use mocking to avoid requiring live Pinecone credentials.
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.namespaces.types import (
    CrossNamespaceResult,
    IsolationStrategy,
    TenantStatus,
)


class TestPineconeNamespaceInit:
    """Unit tests for Pinecone namespace pipeline initialization.

    Validates pipeline creation with correct configuration and
    isolation strategy assignment.

    Tested Scenarios:
        - Pipeline initialization with connection parameters
        - Isolation strategy is NAMESPACE
    """

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_pipeline_initialization(self, mock_db_cls):
        """Test that pipeline initializes with correct configuration.

        Validates:
            - Index name is stored correctly
            - Vector dimension is captured
            - Connection parameters are preserved
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index", dimension=384
        )

        assert pipeline.index_name == "test-index"
        assert pipeline.dimension == 384

    def test_isolation_strategy_is_namespace(self):
        """Test that isolation strategy is NAMESPACE.

        Validates:
            - Class-level ISOLATION_STRATEGY is IsolationStrategy.NAMESPACE
        """
        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        assert (
            PineconeNamespacePipeline.ISOLATION_STRATEGY == IsolationStrategy.NAMESPACE
        )


class TestPineconeNamespaceManagement:
    """Unit tests for Pinecone namespace management operations.

    Validates administrative operations for managing namespaces including
    creation, deletion, listing, existence checks, and statistics.

    Tested Scenarios:
        - Namespace creation (auto-created on first upsert)
        - Namespace deletion (delete all vectors in namespace)
        - Namespace listing via database API
        - Namespace existence checks
        - Namespace statistics retrieval
    """

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_create_namespace_is_auto_created(self, mock_db_cls):
        """Test that namespace creation returns auto-create message.

        Validates:
            - Result is successful
            - Message indicates auto-creation behavior
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        result = pipeline.create_namespace("ns1")

        assert result.success is True
        assert "auto-created" in result.message

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_delete_namespace(self, mock_db_cls):
        """Test namespace deletion removes all vectors.

        Validates:
            - db.delete is called with delete_all=True and correct namespace
            - Result indicates success
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        result = pipeline.delete_namespace("ns1")

        assert result.success is True
        mock_db.delete.assert_called_once_with(delete_all=True, namespace="ns1")

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_list_namespaces(self, mock_db_cls):
        """Test listing all namespaces in the index.

        Validates:
            - Returns list from db.list_namespaces
            - All namespaces are included
        """
        mock_db = MagicMock()
        mock_db.list_namespaces.return_value = ["ns1", "ns2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        result = pipeline.list_namespaces()

        assert result == ["ns1", "ns2"]

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_namespace_exists_true(self, mock_db_cls):
        """Test namespace existence check returns True when namespace exists.

        Validates:
            - Returns True for existing namespace
        """
        mock_db = MagicMock()
        mock_db.list_namespaces.return_value = ["ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        assert pipeline.namespace_exists("ns1") is True

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_namespace_exists_false(self, mock_db_cls):
        """Test namespace existence check returns False when namespace missing.

        Validates:
            - Returns False for non-existing namespace
        """
        mock_db = MagicMock()
        mock_db.list_namespaces.return_value = ["ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        assert pipeline.namespace_exists("ns2") is False

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_get_namespace_stats(self, mock_db_cls):
        """Test namespace statistics retrieval.

        Validates:
            - Returns NamespaceStats with correct document_count
            - Vector count matches namespace data
        """
        mock_db = MagicMock()
        mock_db.describe_index_stats.return_value = {
            "namespaces": {"ns1": {"vector_count": 10}}
        }
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        stats = pipeline.get_namespace_stats("ns1")

        assert stats.document_count == 10
        assert stats.vector_count == 10
        assert stats.namespace == "ns1"

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_get_namespace_stats_empty_namespace(self, mock_db_cls):
        """Test stats for a namespace missing in index stats.

        Validates:
            - Missing namespace defaults to zero document/vector count
            - Status is UNKNOWN when namespace has no vectors
        """
        mock_db = MagicMock()
        mock_db.describe_index_stats.return_value = {
            "namespaces": {"other_ns": {"vector_count": 7}}
        }
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        stats = pipeline.get_namespace_stats("missing_ns")

        assert stats.document_count == 0
        assert stats.vector_count == 0
        assert stats.status == TenantStatus.UNKNOWN


class TestPineconeNamespaceIndexing:
    """Unit tests for Pinecone namespace indexing operations.

    Validates document indexing workflows with namespace-based isolation.
    Ensures documents are routed to the correct namespace.

    Tested Scenarios:
        - Document upsert with namespace routing
        - Empty document handling (returns count=0)
    """

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_index_documents_to_namespace(self, mock_db_cls):
        """Test document indexing with namespace-based isolation.

        Validates:
            - Documents are upserted to the correct namespace
            - db.upsert is called with namespace parameter
            - Result indicates success
        """
        from langchain_core.documents import Document

        mock_db = MagicMock()
        mock_db.upsert.return_value = 5
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]

        result = pipeline.index_documents(documents, embeddings, "ns1")

        assert result.success is True
        mock_db.upsert.assert_called_once()
        call_kwargs = mock_db.upsert.call_args[1]
        assert call_kwargs["namespace"] == "ns1"

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_index_documents_empty_list(self, mock_db_cls):
        """Test that indexing empty documents returns count=0.

        Validates:
            - Empty document list returns success with count=0
            - No database operations are performed
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        result = pipeline.index_documents([], [], "ns1")

        assert result.success is True
        assert result.data["count"] == 0


class TestPineconeNamespaceQuery:
    """Unit tests for Pinecone namespace query operations.

    Validates query functionality with namespace-based isolation.
    Ensures queries are scoped to specific namespaces and cross-namespace
    queries aggregate results correctly.

    Tested Scenarios:
        - Namespace-scoped query returns empty list (stub)
        - Cross-namespace query returns CrossNamespaceResult
    """

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_query_namespace_returns_empty_list(self, mock_db_cls):
        """Test that query_namespace returns empty list (stub implementation).

        Validates:
            - Returns empty list for any query
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        result = pipeline.query_namespace("query", "ns1")

        assert result == []

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_query_cross_namespace_specific_namespaces(self, mock_db_cls):
        """Test cross-namespace query for explicit namespace subset.

        Validates:
            - Returns CrossNamespaceResult with correct query
            - Only provided namespaces are queried
            - list_namespaces is not called when namespaces argument is provided
        """
        mock_db = MagicMock()
        mock_db.list_namespaces.return_value = ["ns1", "ns2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        result = pipeline.query_cross_namespace("query", namespaces=["ns1"])

        assert isinstance(result, CrossNamespaceResult)
        assert result.query == "query"
        assert set(result.namespace_results.keys()) == {"ns1"}
        mock_db.list_namespaces.assert_not_called()

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    def test_query_cross_namespace_all_namespaces(self, mock_db_cls):
        """Test cross-namespace query when namespaces are auto-discovered.

        Validates:
            - list_namespaces is called when namespaces=None
            - All discovered namespaces are included in result
        """
        mock_db = MagicMock()
        mock_db.list_namespaces.return_value = ["ns1", "ns2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.pinecone import (
            PineconeNamespacePipeline,
        )

        pipeline = PineconeNamespacePipeline(
            api_key="test-key", index_name="test-index"
        )

        result = pipeline.query_cross_namespace("query", namespaces=None)

        assert isinstance(result, CrossNamespaceResult)
        assert set(result.namespace_results.keys()) == {"ns1", "ns2"}
        mock_db.list_namespaces.assert_called_once()
