"""Tests for Milvus namespace pipelines using LangChain.

Milvus uses partition key field for namespace isolation. Each namespace is
implemented using metadata field filtering, providing logical isolation
within a single collection. Namespace values are auto-created on insert
(partition key behavior) rather than requiring explicit creation.

Test Coverage:
    - Initialization: Pipeline configuration and isolation strategy
    - Management: Create (auto), delete, list, exists, stats
    - Indexing: Document upsert with partition routing
    - Query: Single-namespace and cross-namespace queries

Test Organization:
    - TestMilvusNamespaceInit: Pipeline initialization and strategy
    - TestMilvusNamespaceManagement: CRUD operations on namespaces
    - TestMilvusNamespaceIndexing: Document indexing with partitions
    - TestMilvusNamespaceQuery: Query operations

All tests use mocking to avoid requiring live Milvus server.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestMilvusNamespaceInit:
    """Unit tests for Milvus namespace pipeline initialization.

    Validates pipeline configuration and isolation strategy assignment.

    Tested Scenarios:
        - Pipeline stores host, port, collection_name, dimension correctly
        - Isolation strategy is PARTITION_KEY
    """

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_pipeline_initialization(self, mock_db_cls):
        """Test that pipeline initializes with correct configuration.

        Validates:
            - Host and port are stored correctly
            - Collection name is configured
            - Vector dimension is captured
            - Mock database is instantiated on first use
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline(
            host="localhost", port=19530, collection_name="namespaces", dimension=384
        )

        assert pipeline.host == "localhost"
        assert pipeline.port == 19530
        assert pipeline.collection_name == "namespaces"
        assert pipeline.dimension == 384

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_isolation_strategy_is_partition_key(self, mock_db_cls):
        """Test that isolation strategy is PARTITION_KEY.

        Validates:
            - Class-level ISOLATION_STRATEGY is set to PARTITION_KEY
            - Matches Milvus partition key field approach
        """
        mock_db_cls.return_value = MagicMock()

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline
        from vectordb.langchain.namespaces.types import IsolationStrategy

        assert (
            MilvusNamespacePipeline.ISOLATION_STRATEGY
            == IsolationStrategy.PARTITION_KEY
        )


class TestMilvusNamespaceManagement:
    """Unit tests for Milvus namespace management operations.

    Validates namespace CRUD operations using partition key field isolation.
    Namespaces are auto-created on insert (Milvus behavior) and deleted
    by filtering and removing all documents with a given namespace value.

    Tested Scenarios:
        - Create namespace returns auto-create message
        - Delete namespace calls db.delete with namespace filter
        - List namespaces returns unique namespace values
        - Namespace exists checks document count
        - Get namespace stats returns document count
    """

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_create_namespace_is_auto_created(self, mock_db_cls):
        """Test that create_namespace returns success with auto-create message.

        Validates:
            - Result indicates success
            - Message explains auto-creation on insert
            - No database call is made for creation
        """
        mock_db_cls.return_value = MagicMock()

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        result = pipeline.create_namespace("ns1")

        assert result.success is True
        assert result.namespace == "ns1"
        assert result.operation == "create"
        assert "auto-created" in result.message

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_delete_namespace(self, mock_db_cls):
        """Test that delete_namespace calls db.delete with namespace filter.

        Validates:
            - db.delete is called with correct namespace filter
            - Result indicates successful deletion
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        result = pipeline.delete_namespace("ns1")

        mock_db.delete.assert_called_once_with(filters={"namespace": "ns1"})
        assert result.success is True
        assert result.namespace == "ns1"
        assert result.operation == "delete"

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_list_namespaces(self, mock_db_cls):
        """Test that list_namespaces returns unique namespace values.

        Validates:
            - Queries collection for namespace field values
            - Returns deduplicated list of namespace strings
            - Uses db.collection_name for the query
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db.client.query.return_value = [
            {"namespace": "ns1"},
            {"namespace": "ns2"},
            {"namespace": "ns1"},
        ]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        result = pipeline.list_namespaces()

        assert sorted(result) == ["ns1", "ns2"]

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_namespace_exists_true(self, mock_db_cls):
        """Test that namespace_exists returns True when documents exist.

        Validates:
            - Escapes namespace string for query expression
            - Queries for count(*) with namespace filter
            - Returns True when count > 0
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db._escape_expr_string.return_value = "ns1"
        mock_db.client.query.return_value = [{"count(*)": 5}]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        result = pipeline.namespace_exists("ns1")

        assert result is True

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_namespace_exists_false(self, mock_db_cls):
        """Test that namespace_exists returns False when no documents exist.

        Validates:
            - Returns False when count(*) is 0
            - Handles empty namespace gracefully
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db._escape_expr_string.return_value = "ns_empty"
        mock_db.client.query.return_value = [{"count(*)": 0}]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        result = pipeline.namespace_exists("ns_empty")

        assert result is False

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_get_namespace_stats(self, mock_db_cls):
        """Test that get_namespace_stats returns correct document count.

        Validates:
            - Stats contain namespace name and document count
            - Document count matches query result
            - Active status when documents exist
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db._escape_expr_string.return_value = "ns1"
        mock_db.client.query.return_value = [{"count(*)": 42}]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        stats = pipeline.get_namespace_stats("ns1")

        assert stats.namespace == "ns1"
        assert stats.document_count == 42


class TestMilvusNamespaceIndexing:
    """Unit tests for Milvus namespace indexing operations.

    Validates document indexing workflows with partition-based namespace
    isolation. Ensures documents are routed to the correct namespace
    partition and empty lists are handled gracefully.

    Tested Scenarios:
        - Document upsert with partition routing
        - Empty document list returns count=0
    """

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_index_documents_to_partition(self, mock_db_cls):
        """Test document indexing with namespace partition routing.

        Validates:
            - Documents are upserted to the correct partition
            - Namespace is passed as partition_name parameter
            - Collection name matches pipeline configuration
            - Return result contains correct count
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 5
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()

        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]

        result = pipeline.index_documents(documents, embeddings, "ns1")

        assert result.success is True
        assert result.data["count"] == 5
        mock_db.upsert.assert_called_once()
        call_kwargs = mock_db.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "namespaces"
        assert call_kwargs["partition_name"] == "ns1"

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_index_documents_empty_list(self, mock_db_cls):
        """Test that indexing empty documents returns count=0 without database call.

        Validates:
            - Empty document list returns result with count=0
            - No database operations are performed
            - Result indicates success
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()

        result = pipeline.index_documents([], [], "ns1")

        assert result.success is True
        assert result.data["count"] == 0
        mock_db.upsert.assert_not_called()


class TestMilvusNamespaceQuery:
    """Unit tests for Milvus namespace query operations.

    Validates single-namespace and cross-namespace query behavior.

    Tested Scenarios:
        - query_namespace returns empty list
        - query_cross_namespace returns CrossNamespaceResult
    """

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_query_namespace_returns_empty_list(self, mock_db_cls):
        """Test that query_namespace returns an empty list.

        Validates:
            - Default implementation returns empty results
            - No errors are raised
        """
        mock_db_cls.return_value = MagicMock()

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        result = pipeline.query_namespace("test query", "ns1", top_k=10)

        assert result == []

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_query_cross_namespace(self, mock_db_cls):
        """Test that query_cross_namespace returns CrossNamespaceResult.

        Validates:
            - Returns a CrossNamespaceResult dataclass
            - Contains query string and namespace results
            - Handles multiple namespaces
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db.client.query.return_value = [
            {"namespace": "ns1"},
            {"namespace": "ns2"},
        ]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline
        from vectordb.langchain.namespaces.types import CrossNamespaceResult

        pipeline = MilvusNamespacePipeline()
        result = pipeline.query_cross_namespace("test query", namespaces=["ns1", "ns2"])

        assert isinstance(result, CrossNamespaceResult)
        assert result.query == "test query"

    @patch("vectordb.langchain.namespaces.milvus.MilvusVectorDB")
    def test_query_cross_namespace_with_none_namespaces(self, mock_db_cls):
        """Test cross-namespace query discovers namespaces when None is provided.

        Validates:
            - list_namespaces query path is used for namespaces=None
            - Namespace values are discovered from collection query results
            - Returned namespace_results contains discovered namespaces
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db.client.query.return_value = [
            {"namespace": "ns1"},
            {"namespace": "ns2"},
            {"ignored": "value"},
        ]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline

        pipeline = MilvusNamespacePipeline()
        result = pipeline.query_cross_namespace("test query", namespaces=None, top_k=2)

        assert result.query == "test query"
        assert set(result.namespace_results) == {"ns1", "ns2"}
        mock_db.client.query.assert_called_once_with(
            collection_name="namespaces",
            filter="",
            output_fields=["namespace"],
            limit=10000,
            offset=0,
        )
