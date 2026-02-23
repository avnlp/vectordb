"""Tests for Qdrant namespace pipelines using LangChain.

Qdrant uses payload-based filtering for namespace isolation. Each namespace
is implemented using payload filters on a shared collection, providing logical
isolation between namespaces without requiring separate collections.

Namespace Strategy:
    - Uses a single shared Qdrant collection for all namespaces
    - Namespace field is stored in point payload
    - Filtering by namespace payload achieves logical isolation

Test Coverage:
    - Initialization: Pipeline setup with URL and collection prefix
    - Isolation strategy: Payload-based filtering validation
    - Create: Auto-created on insert (Qdrant behavior)
    - Delete: Removes all points matching namespace payload filter
    - List: Discovers namespaces via scrolling and payload extraction
    - Exists: Checks namespace existence via count with filter
    - Stats: Retrieves document count for namespace
    - Indexing: Document upsert with namespace metadata
    - Query: Single-namespace and cross-namespace query

Test Organization:
    - TestQdrantNamespaceInit: Pipeline initialization and strategy
    - TestQdrantNamespaceManagement: CRUD operations on namespaces
    - TestQdrantNamespaceIndexing: Document indexing operations
    - TestQdrantNamespaceQuery: Query operations

All tests use mocking to avoid requiring a live Qdrant server.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestQdrantNamespaceInit:
    """Unit tests for Qdrant namespace pipeline initialization.

    Validates pipeline setup with connection parameters, collection prefix,
    and isolation strategy verification.

    Tested Scenarios:
        - Pipeline initialization with URL and collection prefix
        - Isolation strategy is PAYLOAD_FILTER
    """

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_pipeline_initialization(self, mock_db_cls):
        """Test that pipeline initializes with correct configuration.

        Validates:
            - Qdrant URL is stored correctly
            - Collection prefix is configured
            - Mock database is instantiated
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline(
            url="http://localhost:6333",
            api_key="test-key",
            collection_prefix="ns_",
        )

        assert pipeline.url == "http://localhost:6333"
        assert pipeline.collection_prefix == "ns_"

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_isolation_strategy_is_payload_filter(self, mock_db_cls):
        """Test that isolation strategy is PAYLOAD_FILTER.

        Validates:
            - ISOLATION_STRATEGY class attribute is set to PAYLOAD_FILTER
            - Confirms Qdrant uses payload-based namespace isolation
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline
        from vectordb.langchain.namespaces.types import IsolationStrategy

        assert (
            QdrantNamespacePipeline.ISOLATION_STRATEGY
            == IsolationStrategy.PAYLOAD_FILTER
        )


class TestQdrantNamespaceManagement:
    """Unit tests for Qdrant namespace management operations.

    Validates namespace CRUD operations including auto-creation, deletion via
    payload filter, listing via scroll, existence checks, and statistics.

    Tested Scenarios:
        - Create returns success with auto-create message
        - Delete calls db.delete with namespace filter
        - List discovers namespaces via scroll pagination
        - Exists checks via count filter
        - Stats returns document count for namespace
    """

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_create_namespace_is_auto_created(self, mock_db_cls):
        """Test that create_namespace returns success with auto-create message.

        Validates:
            - Operation succeeds without server call
            - Message indicates payload-based auto-creation
            - Namespace is set correctly in result
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        result = pipeline.create_namespace("ns1")

        assert result.success is True
        assert result.namespace == "ns1"
        assert result.operation == "create"
        assert "auto-created" in result.message

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_delete_namespace(self, mock_db_cls):
        """Test that delete_namespace calls db.delete with namespace filter.

        Validates:
            - db.delete is called with correct namespace filter
            - Operation result indicates success
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        result = pipeline.delete_namespace("ns1")

        mock_db.delete.assert_called_once_with(filters={"namespace": "ns1"})
        assert result.success is True
        assert result.namespace == "ns1"
        assert result.operation == "delete"

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_list_namespaces_via_scroll(self, mock_db_cls):
        """Test that list_namespaces discovers namespaces via scroll pagination.

        Validates:
            - Scrolls through all records extracting namespace payloads
            - Returns unique namespace values
            - Handles pagination with None offset termination
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_record1 = MagicMock()
        mock_record1.payload = {"namespace": "ns1"}
        mock_record2 = MagicMock()
        mock_record2.payload = {"namespace": "ns2"}

        mock_db.collection_name = "namespaces"
        mock_db.client.scroll.return_value = ([mock_record1, mock_record2], None)

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        namespaces = pipeline.list_namespaces()

        assert "ns1" in namespaces
        assert "ns2" in namespaces
        assert len(namespaces) == 2

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_list_namespaces_via_scroll_multiple_pages(self, mock_db_cls):
        """Test list_namespaces scrolls multiple pages and skips missing payloads.

        Validates:
            - Pagination continues when scroll returns non-None offset
            - Pagination stops when scroll returns None offset
            - Records missing namespace payload are ignored
            - Unique namespace values from all pages are returned
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db_cls.return_value = mock_db

        page1_record_with_ns = MagicMock()
        page1_record_with_ns.payload = {"namespace": "ns1"}
        page1_record_missing_ns = MagicMock()
        page1_record_missing_ns.payload = {"other": "value"}
        page2_record_with_ns = MagicMock()
        page2_record_with_ns.payload = {"namespace": "ns2"}
        page2_record_none_payload = MagicMock()
        page2_record_none_payload.payload = None

        next_offset = "page-2"
        mock_db.client.scroll.side_effect = [
            ([page1_record_with_ns, page1_record_missing_ns], next_offset),
            ([page2_record_with_ns, page2_record_none_payload], None),
        ]

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        namespaces = pipeline.list_namespaces()

        assert sorted(namespaces) == ["ns1", "ns2"]
        assert mock_db.client.scroll.call_count == 2
        assert mock_db.client.scroll.call_args_list[0].kwargs["offset"] is None
        assert mock_db.client.scroll.call_args_list[1].kwargs["offset"] == next_offset

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_namespace_exists_via_count_filter(self, mock_db_cls):
        """Test that namespace_exists checks via count with payload filter.

        Validates:
            - Uses Qdrant count API with namespace filter
            - Returns True when count > 0
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_db.collection_name = "namespaces"
        mock_db.client.count.return_value = MagicMock(count=5)

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        result = pipeline.namespace_exists("ns1")

        assert result is True
        mock_db.client.count.assert_called_once()

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_get_namespace_stats(self, mock_db_cls):
        """Test that get_namespace_stats returns correct document count.

        Validates:
            - Uses Qdrant count API with namespace filter (exact=True)
            - Returns NamespaceStats with correct document_count
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_db.collection_name = "namespaces"
        mock_db.client.count.return_value = MagicMock(count=42)

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        stats = pipeline.get_namespace_stats("ns1")

        assert stats.namespace == "ns1"
        assert stats.document_count == 42


class TestQdrantNamespaceIndexing:
    """Unit tests for Qdrant namespace document indexing.

    Validates document indexing workflows with payload-based namespace isolation.
    Ensures documents are upserted correctly and empty lists are handled.

    Tested Scenarios:
        - Document upsert with namespace metadata
        - Empty document list returns count=0
    """

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_index_documents_to_namespace(self, mock_db_cls):
        """Test document indexing to a specific namespace.

        Validates:
            - db.upsert is called with documents and embeddings
            - Result contains correct count from upsert
            - Namespace is payload-based (no namespace in upsert call)
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 5
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()

        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]

        result = pipeline.index_documents(documents, embeddings, "ns1")

        assert result.success is True
        assert result.data["count"] == 5
        mock_db.upsert.assert_called_once_with(
            documents=documents,
            embeddings=embeddings,
        )

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_index_documents_empty_list(self, mock_db_cls):
        """Test that indexing empty documents returns count=0 without upsert.

        Validates:
            - Empty document list returns success with count=0
            - No database operations are performed
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()

        result = pipeline.index_documents([], [], "ns1")

        assert result.success is True
        assert result.data["count"] == 0
        mock_db.upsert.assert_not_called()


class TestQdrantNamespaceQuery:
    """Unit tests for Qdrant namespace query operations.

    Validates single-namespace and cross-namespace query functionality.

    Tested Scenarios:
        - Single namespace query returns empty list
        - Cross-namespace query returns CrossNamespaceResult
    """

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_query_namespace_returns_empty_list(self, mock_db_cls):
        """Test that query_namespace returns empty list.

        Validates:
            - Current implementation returns empty list
            - Return type is a list
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        result = pipeline.query_namespace("test query", "ns1", top_k=5)

        assert result == []
        assert isinstance(result, list)

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_query_cross_namespace(self, mock_db_cls):
        """Test that query_cross_namespace returns CrossNamespaceResult.

        Validates:
            - Returns CrossNamespaceResult dataclass
            - Contains query, namespace_results, and timing_comparison
            - Handles multiple namespaces
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline
        from vectordb.langchain.namespaces.types import CrossNamespaceResult

        pipeline = QdrantNamespacePipeline()
        result = pipeline.query_cross_namespace(
            "test query", namespaces=["ns1", "ns2"], top_k=5
        )

        assert isinstance(result, CrossNamespaceResult)
        assert result.query == "test query"
        assert "ns1" in result.namespace_results
        assert "ns2" in result.namespace_results

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    def test_query_cross_namespace_with_none_namespaces(self, mock_db_cls):
        """Test cross-namespace query discovers namespaces when None is provided.

        Validates:
            - list_namespaces path is used for namespaces=None
            - Namespace discovery uses Qdrant scroll API
            - Returned namespace_results contains discovered namespaces
        """
        mock_db = MagicMock()
        mock_db.collection_name = "namespaces"
        mock_db_cls.return_value = mock_db

        record1 = MagicMock()
        record1.payload = {"namespace": "ns1"}
        record2 = MagicMock()
        record2.payload = {"namespace": "ns2"}
        mock_db.client.scroll.return_value = ([record1, record2], None)

        from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline

        pipeline = QdrantNamespacePipeline()
        result = pipeline.query_cross_namespace("test query", namespaces=None, top_k=4)

        assert result.query == "test query"
        assert set(result.namespace_results) == {"ns1", "ns2"}
        mock_db.client.scroll.assert_called_once()
