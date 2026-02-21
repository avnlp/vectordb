"""Tests for Weaviate namespace pipelines using LangChain.

Weaviate uses native multi-tenancy (tenant mechanism) for namespace isolation.
Each namespace maps to a Weaviate tenant, providing strong isolation with
efficient vector operations. The pipeline supports create, delete, list,
exists, stats, indexing, and cross-namespace query operations.

Test Coverage:
    - Initialization: Pipeline creation with URL, API key, and prefix
    - Isolation strategy: Verifies TENANT strategy is used
    - Namespace management: Create, delete, list, exists, stats
    - Indexing: Document upsert to tenant-specific namespaces
    - Query: Single-namespace and cross-namespace queries

Test Organization:
    - TestWeaviateNamespaceInit: Pipeline initialization and strategy
    - TestWeaviateNamespaceManagement: CRUD operations on namespaces
    - TestWeaviateNamespaceIndexing: Document indexing to tenants
    - TestWeaviateNamespaceQuery: Namespace-scoped queries

All tests use mocking to avoid requiring a live Weaviate server.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestWeaviateNamespaceInit:
    """Unit tests for Weaviate namespace pipeline initialization.

    Validates pipeline creation with connection parameters and that the
    correct isolation strategy (TENANT) is configured.
    """

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_pipeline_initialization(self, mock_db_cls: MagicMock) -> None:
        """Test that pipeline initializes with correct configuration."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(
            url="http://localhost:8080",
            collection_prefix="ns_",
        )

        assert pipeline.url == "http://localhost:8080"
        assert pipeline.collection_prefix == "ns_"

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_isolation_strategy_is_tenant(self, mock_db_cls: MagicMock) -> None:
        """Test that Weaviate uses TENANT isolation strategy."""
        mock_db_cls.return_value = MagicMock()

        from vectordb.langchain.namespaces.types import IsolationStrategy
        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        assert WeaviateNamespacePipeline.ISOLATION_STRATEGY == IsolationStrategy.TENANT


class TestWeaviateNamespaceManagement:
    """Unit tests for Weaviate namespace management operations.

    Validates create, delete, list, exists, and stats operations
    using Weaviate's tenant mechanism for namespace isolation.
    """

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_create_namespace(self, mock_db_cls: MagicMock) -> None:
        """Test namespace creation calls db.create_tenant."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        result = pipeline.create_namespace("ns1")

        mock_db.create_tenant.assert_called_once_with(tenant="ns1")
        assert result.success is True

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_delete_namespace(self, mock_db_cls: MagicMock) -> None:
        """Test namespace deletion calls db.delete_tenant."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        result = pipeline.delete_namespace("ns1")

        mock_db.delete_tenant.assert_called_once_with(tenant="ns1")
        assert result.success is True

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_list_namespaces(self, mock_db_cls: MagicMock) -> None:
        """Test listing namespaces returns tenant list."""
        mock_db = MagicMock()
        mock_db.list_tenants.return_value = ["ns1", "ns2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        result = pipeline.list_namespaces()

        assert result == ["ns1", "ns2"]

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_namespace_exists_true(self, mock_db_cls: MagicMock) -> None:
        """Test namespace_exists returns True for existing tenant."""
        mock_db = MagicMock()
        mock_db.list_tenants.return_value = ["ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")

        assert pipeline.namespace_exists("ns1") is True

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_namespace_exists_false(self, mock_db_cls: MagicMock) -> None:
        """Test namespace_exists returns False for missing tenant."""
        mock_db = MagicMock()
        mock_db.list_tenants.return_value = ["ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")

        assert pipeline.namespace_exists("ns2") is False

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_get_namespace_stats(self, mock_db_cls: MagicMock) -> None:
        """Test namespace stats retrieves document count from tenant."""
        mock_db = MagicMock()
        mock_db.collection.aggregate.over_all.return_value = MagicMock(total_count=10)
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        stats = pipeline.get_namespace_stats("ns1")

        mock_db.with_tenant.assert_called_once_with("ns1")
        assert stats.document_count == 10


class TestWeaviateNamespaceIndexing:
    """Unit tests for Weaviate namespace document indexing.

    Validates document upsert to tenant-specific namespaces and
    empty document handling.
    """

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_index_documents_to_tenant(self, mock_db_cls: MagicMock) -> None:
        """Test document indexing calls upsert with correct tenant."""
        mock_db = MagicMock()
        mock_db.upsert.return_value = 5
        mock_db.list_tenants.return_value = ["ns1"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]

        result = pipeline.index_documents(documents, embeddings, "ns1")

        mock_db.upsert.assert_called_once_with(
            documents=documents,
            embeddings=embeddings,
            tenant="ns1",
        )
        assert result.data["count"] == 5

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_index_documents_creates_tenant_if_missing(
        self, mock_db_cls: MagicMock
    ) -> None:
        """Test indexing creates tenant when namespace does not exist.

        Validates:
            - namespace_exists checks tenant list
            - create_tenant is called when namespace is missing
            - upsert still runs for requested tenant
        """
        mock_db = MagicMock()
        mock_db.list_tenants.return_value = []
        mock_db.upsert.return_value = 1
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        documents = [Document(page_content="doc1", metadata={"id": "1"})]
        embeddings = [[0.1] * 384]

        result = pipeline.index_documents(documents, embeddings, "ns_missing")

        mock_db.create_tenant.assert_called_once_with(tenant="ns_missing")
        mock_db.upsert.assert_called_once_with(
            documents=documents,
            embeddings=embeddings,
            tenant="ns_missing",
        )
        assert result.data["count"] == 1

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_index_documents_empty_list(self, mock_db_cls: MagicMock) -> None:
        """Test indexing empty documents returns count 0 without db call."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        result = pipeline.index_documents([], [], "ns1")

        assert result.data["count"] == 0
        mock_db.upsert.assert_not_called()


class TestWeaviateNamespaceQuery:
    """Unit tests for Weaviate namespace query operations.

    Validates single-namespace and cross-namespace query behavior.
    """

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_query_namespace_returns_empty_list(self, mock_db_cls: MagicMock) -> None:
        """Test query_namespace returns empty list."""
        mock_db_cls.return_value = MagicMock()

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        result = pipeline.query_namespace("test query", "ns1", top_k=5)

        assert result == []

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_query_cross_namespace(self, mock_db_cls: MagicMock) -> None:
        """Test cross-namespace query returns CrossNamespaceResult."""
        mock_db = MagicMock()
        mock_db.list_tenants.return_value = ["ns1", "ns2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.types import CrossNamespaceResult
        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        result = pipeline.query_cross_namespace("test query")

        assert isinstance(result, CrossNamespaceResult)

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    def test_query_cross_namespace_with_explicit_namespaces(
        self, mock_db_cls: MagicMock
    ) -> None:
        """Test explicit namespaces path does not require tenant discovery.

        Validates:
            - Provided namespaces are used directly
            - list_tenants is not called when namespaces argument is explicit
            - Returned namespace_results contains provided namespaces
        """
        mock_db = MagicMock()
        mock_db.list_tenants.return_value = ["unused"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.namespaces.weaviate import (
            WeaviateNamespacePipeline,
        )

        pipeline = WeaviateNamespacePipeline(url="http://localhost:8080")
        result = pipeline.query_cross_namespace(
            "test query", namespaces=["ns1", "ns2"], top_k=3
        )

        assert set(result.namespace_results) == {"ns1", "ns2"}
        mock_db.list_tenants.assert_not_called()
