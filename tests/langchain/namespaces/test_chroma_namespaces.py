"""Tests for Chroma namespace pipelines using LangChain.

Chroma uses collection-per-namespace pattern. Each namespace maps to a
separate Chroma collection with a configurable prefix (default: "ns_"),
providing complete data isolation at the collection level.

Test Coverage:
    - Initialization: Pipeline setup with path and prefix
    - Collection naming: Prefix-based name generation
    - Namespace management: Create, delete, list, exists, stats
    - Document indexing: Upsert to namespace-specific collections
    - Query: Single-namespace and cross-namespace queries

Test Organization:
    - TestChromaNamespaceInit: Pipeline initialization and configuration
    - TestChromaNamespaceManagement: CRUD operations on namespaces
    - TestChromaNamespaceIndexing: Document indexing operations
    - TestChromaNamespaceQuery: Query operations

All tests use mocking to avoid requiring actual Chroma database.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestChromaNamespaceInit:
    """Unit tests for ChromaNamespacePipeline initialization.

    Validates pipeline configuration, isolation strategy, and
    collection name generation with configurable prefix.

    Tested Scenarios:
        - Pipeline initialization with persistence path and prefix
        - Isolation strategy is COLLECTION
        - Collection name generation with prefix
    """

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_pipeline_initialization(self, mock_db_cls):
        """Test that pipeline initializes with correct configuration.

        Validates:
            - Persistence path is stored correctly
            - Collection prefix is configured
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline(
            path="./test_chroma_data", collection_prefix="ns_"
        )

        assert pipeline.path == "./test_chroma_data"
        assert pipeline.collection_prefix == "ns_"

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_isolation_strategy_is_collection(self, mock_db_cls):
        """Test that isolation strategy is COLLECTION.

        Validates:
            - ISOLATION_STRATEGY class attribute is IsolationStrategy.COLLECTION
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline
        from vectordb.langchain.namespaces.types import IsolationStrategy

        assert (
            ChromaNamespacePipeline.ISOLATION_STRATEGY == IsolationStrategy.COLLECTION
        )

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_get_collection_name(self, mock_db_cls):
        """Test collection name generation with configured prefix.

        Validates:
            - Namespace is combined with prefix to form collection name
            - Prefix is prepended correctly to namespace identifier
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()

        collection_name = pipeline._get_collection_name("ns1")
        assert collection_name == "ns_ns1"


class TestChromaNamespaceManagement:
    """Unit tests for ChromaNamespacePipeline namespace management.

    Validates namespace CRUD operations including creation, deletion,
    listing, existence checks, and statistics retrieval.

    Tested Scenarios:
        - Namespace creation via collection creation
        - Namespace deletion via collection deletion
        - Listing namespaces filtered by prefix
        - Namespace existence check (true and false)
        - Namespace statistics retrieval
    """

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_create_namespace(self, mock_db_cls):
        """Test namespace creation creates a collection.

        Validates:
            - db.create_collection is called with prefixed collection name
            - Result indicates successful creation
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()
        result = pipeline.create_namespace("ns1")

        mock_db.create_collection.assert_called_once_with(collection_name="ns_ns1")
        assert result.success is True

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_delete_namespace(self, mock_db_cls):
        """Test namespace deletion deletes the collection.

        Validates:
            - db.delete_collection is called with prefixed collection name
            - Result indicates successful deletion
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()
        result = pipeline.delete_namespace("ns1")

        mock_db.delete_collection.assert_called_once_with(collection_name="ns_ns1")
        assert result.success is True

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_list_namespaces(self, mock_db_cls):
        """Test listing namespaces filters by prefix.

        Validates:
            - Only collections matching the prefix are returned
            - Prefix is stripped from returned namespace names
            - Collections without prefix are excluded
        """
        mock_db = MagicMock()
        mock_db.list_collections.return_value = ["ns_ns1", "ns_ns2", "other"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()
        namespaces = pipeline.list_namespaces()

        assert namespaces == ["ns1", "ns2"]

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_list_namespaces_ignores_unrelated_collections(self, mock_db_cls):
        """Test listing namespaces excludes collections without namespace prefix.

        Validates:
            - Collections that do not start with prefix are ignored
            - Returned list contains only namespace-derived names
        """
        mock_db = MagicMock()
        mock_db.list_collections.return_value = [
            "ns_train",
            "metrics",
            "tmp",
            "ns_eval",
        ]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()
        namespaces = pipeline.list_namespaces()

        assert namespaces == ["train", "eval"]

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_namespace_exists_true(self, mock_db_cls):
        """Test namespace existence check returns True when collection exists.

        Validates:
            - Returns True when prefixed collection name is in list
        """
        mock_db = MagicMock()
        mock_db.list_collections.return_value = ["ns_ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()

        assert pipeline.namespace_exists("ns1") is True

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_namespace_exists_false(self, mock_db_cls):
        """Test namespace existence check returns False when collection missing.

        Validates:
            - Returns False when prefixed collection name is not in list
        """
        mock_db = MagicMock()
        mock_db.list_collections.return_value = ["ns_ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()

        assert pipeline.namespace_exists("ns2") is False

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_get_namespace_stats(self, mock_db_cls):
        """Test namespace statistics retrieval.

        Validates:
            - db._get_collection is called with prefixed collection name
            - Document count matches collection count
            - Stats namespace field matches input
        """
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 15
        mock_db._get_collection.return_value = mock_collection
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()
        stats = pipeline.get_namespace_stats("ns1")

        mock_db._get_collection.assert_called_once_with("ns_ns1")
        assert stats.document_count == 15
        assert stats.vector_count == 15
        assert stats.namespace == "ns1"


class TestChromaNamespaceIndexing:
    """Unit tests for ChromaNamespacePipeline document indexing.

    Validates document indexing workflows with collection-per-namespace
    isolation. Ensures documents are routed to the correct namespace
    collection.

    Tested Scenarios:
        - Document upsert to namespace-specific collection
        - Empty document list handling
    """

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_index_documents_to_collection(self, mock_db_cls):
        """Test document indexing to namespace-specific collection.

        Validates:
            - Documents are upserted to the correct namespace collection
            - Collection name includes configured prefix
            - Return result contains correct count
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 5
        mock_db.list_collections.return_value = ["ns_ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()

        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]

        result = pipeline.index_documents(documents, embeddings, "ns1")

        mock_db.upsert.assert_called_once_with(
            documents=documents,
            embeddings=embeddings,
            collection_name="ns_ns1",
        )
        assert result.data["count"] == 5

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_index_documents_creates_namespace_if_missing(self, mock_db_cls):
        """Test indexing auto-creates namespace collection when missing.

        Validates:
            - namespace_exists check fails for missing namespace
            - create_namespace is called before upsert
            - Upsert still targets namespace collection
        """
        mock_db = MagicMock()
        mock_db.list_collections.return_value = []
        mock_db.upsert.return_value = 2
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()
        documents = [Document(page_content="doc1", metadata={"id": "1"})]
        embeddings = [[0.1] * 384]

        result = pipeline.index_documents(documents, embeddings, "ns_new")

        mock_db.create_collection.assert_called_once_with(collection_name="ns_ns_new")
        mock_db.upsert.assert_called_once_with(
            documents=documents,
            embeddings=embeddings,
            collection_name="ns_ns_new",
        )
        assert result.data["count"] == 2

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_index_documents_empty_list(self, mock_db_cls):
        """Test that indexing empty documents returns result with count 0.

        Validates:
            - Empty document list returns result immediately
            - Result data contains count of 0
            - No database upsert is performed
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()

        result = pipeline.index_documents([], [], "ns1")

        assert result.data["count"] == 0
        mock_db.upsert.assert_not_called()


class TestChromaNamespaceQuery:
    """Unit tests for ChromaNamespacePipeline query operations.

    Validates single-namespace and cross-namespace query functionality.

    Tested Scenarios:
        - Single namespace query returns empty list (stub)
        - Cross-namespace query returns CrossNamespaceResult
    """

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_query_namespace_returns_empty_list(self, mock_db_cls):
        """Test that query_namespace returns empty list.

        Validates:
            - query_namespace returns an empty list (stub implementation)
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()

        result = pipeline.query_namespace("test query", "ns1", top_k=5)

        assert result == []

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_query_cross_namespace(self, mock_db_cls):
        """Test cross-namespace query returns CrossNamespaceResult.

        Validates:
            - Returns CrossNamespaceResult dataclass
            - Result contains query string
            - Result contains namespace_results dictionary
        """
        mock_db = MagicMock()
        mock_db.list_collections.return_value = ["ns_ns1", "ns_ns2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline
        from vectordb.langchain.namespaces.types import CrossNamespaceResult

        pipeline = ChromaNamespacePipeline()

        result = pipeline.query_cross_namespace("test query", namespaces=["ns1", "ns2"])

        assert isinstance(result, CrossNamespaceResult)
        assert result.query == "test query"
        assert "ns1" in result.namespace_results
        assert "ns2" in result.namespace_results

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    def test_query_cross_namespace_with_none_namespaces(self, mock_db_cls):
        """Test cross-namespace query discovers namespaces when None is provided.

        Validates:
            - list_namespaces path is used when namespaces=None
            - Namespaces are discovered from prefixed collection names
            - Returned namespace_results contains discovered namespaces
        """
        mock_db = MagicMock()
        mock_db.list_collections.return_value = ["ns_ns1", "other", "ns_ns2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline

        pipeline = ChromaNamespacePipeline()
        result = pipeline.query_cross_namespace("test query", namespaces=None, top_k=3)

        assert result.query == "test query"
        assert set(result.namespace_results) == {"ns1", "ns2"}
        mock_db.list_collections.assert_called_once_with()
