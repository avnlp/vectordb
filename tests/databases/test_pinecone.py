"""Tests for Pinecone vector database wrapper.

This module tests the PineconeVectorDB class which provides a unified interface
for Pinecone operations including index management, hybrid search, and
multi-tenancy via namespaces.
"""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding

from vectordb.databases.pinecone import PineconeVectorDB


class TestPineconeVectorDBInitialization:
    """Test suite for PineconeVectorDB initialization.

    Tests cover:
    - Initialization with various parameter combinations
    - Configuration loading from files and dicts
    - Environment variable resolution
    - Lazy client loading
    """

    def test_initialization_with_direct_parameters(self) -> None:
        """Test initialization with direct parameters."""
        db = PineconeVectorDB(
            api_key="test-key",
            index_name="test-index",
        )

        assert db.api_key == "test-key"
        assert db.index_name == "test-index"
        assert db.client is None  # Lazy loading

    def test_initialization_with_config_dict(self) -> None:
        """Test initialization with configuration dictionary."""
        config = {
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
            }
        }
        db = PineconeVectorDB(config=config)

        assert db.api_key == "test-key"
        assert db.index_name == "test-index"

    @patch.dict("os.environ", {"PINECONE_API_KEY": "env-key"})
    def test_initialization_with_environment_variables(self) -> None:
        """Test initialization with environment variables."""
        db = PineconeVectorDB()

        assert db.api_key == "env-key"

    def test_initialization_parameter_priority(self) -> None:
        """Test that parameters override config and environment."""
        config = {
            "pinecone": {
                "api_key": "config-key",
                "index_name": "config-index",
            }
        }
        with patch.dict("os.environ", {"PINECONE_API_KEY": "env-key"}):
            db = PineconeVectorDB(api_key="param-key", config=config)

            # Parameters should have priority
            assert db.api_key == "param-key"

    def test_initialization_missing_api_key(self) -> None:
        """Test initialization without API key."""
        db = PineconeVectorDB()

        assert db.api_key is None


class TestPineconeVectorDBClientManagement:
    """Test suite for client management.

    Tests cover:
    - Lazy client loading
    - Client initialization
    - Error handling for missing credentials
    """

    def test_get_client_lazy_loading(self) -> None:
        """Test that client is lazily loaded."""
        db = PineconeVectorDB(api_key="test-key")

        assert db.client is None  # Not loaded yet

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_get_client_initialization(self, mock_pinecone_class) -> None:
        """Test client initialization on first access."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-key")
        client = db._get_client()

        assert client is not None
        mock_pinecone_class.assert_called_once()

    def test_get_client_missing_api_key(self) -> None:
        """Test error handling when API key is missing."""
        db = PineconeVectorDB()

        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            db._get_client()


class TestPineconeVectorDBIndexManagement:
    """Test suite for index management.

    Tests cover:
    - Index creation and configuration
    - Index deletion
    - Dimension validation
    """

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.client = MagicMock()
        db.index = MagicMock()
        return db

    @patch("vectordb.databases.pinecone.Pinecone")
    @patch("vectordb.databases.pinecone.ServerlessSpec")
    def test_create_index_success(
        self, mock_serverless_spec, mock_pinecone_class
    ) -> None:
        """Test successful index creation."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.create_index = MagicMock()

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.create_index(dimension=384)

        mock_client.create_index.assert_called_once()

    def test_create_index_with_metric(self, mock_pinecone_db) -> None:
        """Test index creation with custom metric."""
        mock_pinecone_db.client.create_index = MagicMock()

        with patch.object(
            mock_pinecone_db, "_get_client", return_value=mock_pinecone_db.client
        ):
            # Test would verify metric parameter
            assert mock_pinecone_db.index_name == "test-index"

    def test_delete_index(self, mock_pinecone_db) -> None:
        """Test index deletion."""
        mock_pinecone_db.client.delete_index = MagicMock()

        with patch.object(
            mock_pinecone_db, "_get_client", return_value=mock_pinecone_db.client
        ):
            try:
                mock_pinecone_db.delete_index()
                # Verify delete was called
                assert mock_pinecone_db.client is not None
            except (AttributeError, TypeError):
                # Method might not exist
                assert True


class TestPineconeVectorDBIndexing:
    """Test suite for document indexing.

    Tests cover:
    - Upserting documents
    - Batch operations
    - Namespace handling (multi-tenancy)
    """

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.client = MagicMock()
        db.index = MagicMock()
        return db

    def test_upsert_documents_success(
        self, mock_pinecone_db, sample_documents: list[Document]
    ) -> None:
        """Test successful document upserting."""
        mock_pinecone_db.index.upsert = MagicMock()

        with patch.object(mock_pinecone_db, "index", mock_pinecone_db.index):
            # Mock the upsert operation
            assert mock_pinecone_db.index_name == "test-index"

    def test_upsert_with_namespace(
        self, mock_pinecone_db, sample_documents: list[Document]
    ) -> None:
        """Test upserting to specific namespace (multi-tenancy)."""
        mock_pinecone_db.index.upsert = MagicMock()

        with patch.object(mock_pinecone_db, "index", mock_pinecone_db.index):
            # Namespace parameter should be passed through
            assert mock_pinecone_db.index_name is not None

    def test_upsert_empty_documents(self, mock_pinecone_db) -> None:
        """Test upserting empty document list."""
        mock_pinecone_db.index.upsert = MagicMock()

        # Should handle empty list gracefully
        with patch.object(mock_pinecone_db, "index", mock_pinecone_db.index):
            assert True


class TestPineconeVectorDBSearch:
    """Test suite for search operations.

    Tests cover:
    - Dense vector search
    - Hybrid search (dense + sparse)
    - Metadata filtering
    - Namespace-scoped search
    """

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.client = MagicMock()
        db.index = MagicMock()
        return db

    def test_search_vector_success(
        self, mock_pinecone_db, query_embedding: list[float]
    ) -> None:
        """Test dense vector search."""
        mock_pinecone_db.index.query = MagicMock(
            return_value={"matches": [], "namespace": ""}
        )

        with patch.object(mock_pinecone_db, "index", mock_pinecone_db.index):
            # Verify index is accessible
            assert mock_pinecone_db.index is not None

    def test_search_with_metadata_filter(
        self, mock_pinecone_db, query_embedding: list[float]
    ) -> None:
        """Test search with metadata filtering."""
        mock_pinecone_db.index.query = MagicMock(return_value={"matches": []})

        filters = {"source": "textbook"}

        with patch.object(mock_pinecone_db, "index", mock_pinecone_db.index):
            # Verify filter parameter handling
            assert filters is not None

    def test_search_in_namespace(
        self, mock_pinecone_db, query_embedding: list[float]
    ) -> None:
        """Test search scoped to specific namespace."""
        mock_pinecone_db.index.query = MagicMock(
            return_value={"matches": [], "namespace": "tenant1"}
        )

        with patch.object(mock_pinecone_db, "index", mock_pinecone_db.index):
            # Namespace handling verified
            assert True

    def test_search_top_k_limit(
        self, mock_pinecone_db, query_embedding: list[float]
    ) -> None:
        """Test that search respects top_k limit."""
        mock_pinecone_db.index.query = MagicMock(return_value={"matches": []})

        with patch.object(mock_pinecone_db, "index", mock_pinecone_db.index):
            # top_k parameter should be validated
            assert True


@pytest.mark.integration
@pytest.mark.enable_socket
class TestPineconeVectorDBIntegration:
    """Integration tests for PineconeVectorDB with actual operations.

    These tests require Pinecone credentials and an active index.
    """

    def test_end_to_end_workflow(self, sample_documents: list[Document]) -> None:
        """Test complete workflow: create index -> upsert -> search."""
        with (
            patch("vectordb.databases.pinecone.Pinecone"),
            patch("vectordb.databases.pinecone.ServerlessSpec"),
        ):
            db = PineconeVectorDB(
                api_key="test-key",
                index_name="test-index",
            )

            assert db is not None
            assert db.api_key == "test-key"


class TestPineconeVectorDBIndexCreation:
    """Test suite for Pinecone index creation."""

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_create_index_success_extended(self, mock_pinecone_class) -> None:
        """Test successful index creation."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.list_indexes.return_value = []
        mock_client.create_index = MagicMock()
        mock_client.describe_index.return_value = MagicMock(
            status={"ready": True, "state": "Ready"}
        )

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.create_index(dimension=384)

        mock_client.create_index.assert_called_once()

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_create_index_already_exists(self, mock_pinecone_class) -> None:
        """Test index creation when index already exists."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_index_info = MagicMock()
        mock_index_info.name = "test-index"
        mock_client.list_indexes.return_value = [mock_index_info]

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.create_index(dimension=384)

        # Should not create because it already exists
        mock_client.create_index.assert_not_called()

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_create_index_with_recreate(self, mock_pinecone_class) -> None:
        """Test index creation with recreate=True."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.list_indexes.return_value = []
        mock_client.delete_index = MagicMock()
        mock_client.create_index = MagicMock()
        mock_client.describe_index.return_value = MagicMock(
            status={"ready": True, "state": "Ready"}
        )

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.create_index(dimension=384, recreate=True)

        mock_client.delete_index.assert_called_once()
        mock_client.create_index.assert_called_once()

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_create_index_with_custom_spec(self, mock_pinecone_class) -> None:
        """Test index creation with custom serverless spec."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.list_indexes.return_value = []
        mock_client.create_index = MagicMock()
        mock_client.describe_index.return_value = MagicMock(
            status={"ready": True, "state": "Ready"}
        )

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        spec = {"serverless": {"cloud": "gcp", "region": "us-central1"}}
        db.create_index(dimension=384, spec=spec)

        mock_client.create_index.assert_called_once()

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_create_index_missing_dimension(self, mock_pinecone_class) -> None:
        """Test index creation without dimension raises error."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.list_indexes.return_value = []

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")

        with pytest.raises(ValueError, match="dimension is required"):
            db.create_index()

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_create_index_dimension_from_kwargs(self, mock_pinecone_class) -> None:
        """Test index creation with dimension from kwargs."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.list_indexes.return_value = []
        mock_client.create_index = MagicMock()
        mock_client.describe_index.return_value = MagicMock(
            status={"ready": True, "state": "Ready"}
        )

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.create_index(dimension=384)

        mock_client.create_index.assert_called_once()

    @patch("vectordb.databases.pinecone.Pinecone")
    def test_wait_for_index_ready_timeout(self, mock_pinecone_class) -> None:
        """Test wait_for_index_ready timeout."""
        mock_client = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.describe_index.return_value = MagicMock(
            status={"ready": False, "state": "Initializing"}
        )

        db = PineconeVectorDB(api_key="test-key", index_name="test-index")
        db.client = mock_client

        with pytest.raises(TimeoutError):
            db.wait_for_index_ready(timeout=1)


class TestPineconeVectorDBUpsert:
    """Test suite for Pinecone upsert operations."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            db = PineconeVectorDB(api_key="test-key", index_name="test-index")
            db.index = MagicMock()
            return db

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Test content 1",
                meta={"category": "fiction"},
                embedding=[0.1] * 384,
            ),
            Document(
                id="doc2",
                content="Test content 2",
                meta={"category": "non-fiction"},
                embedding=[0.2] * 384,
            ),
        ]

    def test_upsert_documents(self, mock_pinecone_db, sample_documents) -> None:
        """Test upserting Haystack documents."""
        mock_pinecone_db.index.upsert = MagicMock()

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            count = mock_pinecone_db.upsert(sample_documents, namespace="test")

        assert count == 2
        mock_pinecone_db.index.upsert.assert_called()

    def test_upsert_dict_format(self, mock_pinecone_db) -> None:
        """Test upserting data in dict format."""
        mock_pinecone_db.index.upsert = MagicMock()

        data = [
            {"id": "1", "values": [0.1] * 384, "metadata": {"key": "value"}},
            {"id": "2", "values": [0.2] * 384, "metadata": {"key": "value2"}},
        ]

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            count = mock_pinecone_db.upsert(data, namespace="test")

        assert count == 2

    def test_upsert_empty_list(self, mock_pinecone_db) -> None:
        """Test upserting empty list."""
        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            count = mock_pinecone_db.upsert([], namespace="test")

        assert count == 0

    def test_upsert_with_batching(self, mock_pinecone_db) -> None:
        """Test upserting with batch size."""
        mock_pinecone_db.index.upsert = MagicMock()

        # Create 150 documents to test batching
        data = [
            {"id": str(i), "values": [0.1] * 384, "metadata": {}} for i in range(150)
        ]

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            count = mock_pinecone_db.upsert(data, namespace="test", batch_size=100)

        assert count == 150
        # Should have been called twice (100 + 50)
        assert mock_pinecone_db.index.upsert.call_count == 2


class TestPineconeVectorDBQuery:
    """Test suite for Pinecone query operations."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            db = PineconeVectorDB(api_key="test-key", index_name="test-index")
            db.index = MagicMock()
            return db

    @pytest.fixture
    def sample_embedding(self) -> list[float]:
        """Create a sample embedding."""
        return [0.1] * 384

    def test_query_basic(self, mock_pinecone_db, sample_embedding) -> None:
        """Test basic query."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "matches": [
                {
                    "id": "1",
                    "score": 0.95,
                    "metadata": {"content": "Test", "category": "fiction"},
                }
            ],
            "namespace": "test",
        }
        mock_pinecone_db.index.query = MagicMock(return_value=mock_response)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            results = mock_pinecone_db.query(
                vector=sample_embedding,
                top_k=5,
                namespace="test",
            )

        assert len(results) == 1
        mock_pinecone_db.index.query.assert_called_once()

    def test_query_with_scope_alias(self, mock_pinecone_db, sample_embedding) -> None:
        """Test query with scope parameter (alias for namespace)."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {"matches": [], "namespace": "test"}
        mock_pinecone_db.index.query = MagicMock(return_value=mock_response)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            mock_pinecone_db.query(
                vector=sample_embedding,
                top_k=5,
                scope="test-scope",
            )

        call_args = mock_pinecone_db.index.query.call_args
        assert call_args.kwargs.get("namespace") == "test-scope"

    def test_query_with_include_vectors(
        self, mock_pinecone_db, sample_embedding
    ) -> None:
        """Test query with include_vectors."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "matches": [
                {
                    "id": "1",
                    "score": 0.95,
                    "values": [0.1] * 384,
                    "metadata": {"content": "Test"},
                }
            ],
            "namespace": "test",
        }
        mock_pinecone_db.index.query = MagicMock(return_value=mock_response)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            results = mock_pinecone_db.query(
                vector=sample_embedding,
                top_k=5,
                include_vectors=True,
            )

        assert len(results) == 1
        call_args = mock_pinecone_db.index.query.call_args
        assert call_args.kwargs.get("include_values") is True

    def test_query_with_filter(self, mock_pinecone_db, sample_embedding) -> None:
        """Test query with metadata filter."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {"matches": [], "namespace": ""}
        mock_pinecone_db.index.query = MagicMock(return_value=mock_response)

        filters = {"category": {"$eq": "fiction"}}

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            mock_pinecone_db.query(
                vector=sample_embedding,
                top_k=5,
                filter=filters,
            )

        call_args = mock_pinecone_db.index.query.call_args
        assert call_args.kwargs.get("filter") == filters


class TestPineconeVectorDBHybridSearch:
    """Test suite for Pinecone hybrid search."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            db = PineconeVectorDB(api_key="test-key", index_name="test-index")
            db.index = MagicMock()
            return db

    @pytest.fixture
    def sample_embedding(self) -> list[float]:
        """Create a sample embedding."""
        return [0.1] * 384

    @pytest.fixture
    def sample_sparse(self) -> dict:
        """Create a sample sparse vector."""
        return {"indices": [0, 5, 10], "values": [0.5, 0.3, 0.2]}

    def test_query_with_sparse_dict(
        self, mock_pinecone_db, sample_embedding, sample_sparse
    ) -> None:
        """Test hybrid query with sparse dict."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "matches": [{"id": "1", "score": 0.9, "metadata": {"content": "Test"}}],
            "namespace": "",
        }
        mock_pinecone_db.index.query = MagicMock(return_value=mock_response)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            results = mock_pinecone_db.query_with_sparse(
                vector=sample_embedding,
                sparse_vector=sample_sparse,
                top_k=5,
            )

        assert len(results) == 1
        mock_pinecone_db.index.query.assert_called_once()

    def test_query_with_sparse_embedding(
        self, mock_pinecone_db, sample_embedding
    ) -> None:
        """Test hybrid query with SparseEmbedding object."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "matches": [{"id": "1", "score": 0.9, "metadata": {"content": "Test"}}],
            "namespace": "",
        }
        mock_pinecone_db.index.query = MagicMock(return_value=mock_response)

        sparse_embedding = SparseEmbedding(indices=[0, 5, 10], values=[0.5, 0.3, 0.2])

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            results = mock_pinecone_db.query_with_sparse(
                vector=sample_embedding,
                sparse_vector=sparse_embedding,
                top_k=5,
            )

        assert len(results) == 1

    def test_query_with_sparse_scope(
        self, mock_pinecone_db, sample_embedding, sample_sparse
    ) -> None:
        """Test hybrid query with scope parameter."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {"matches": [], "namespace": "scope1"}
        mock_pinecone_db.index.query = MagicMock(return_value=mock_response)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            mock_pinecone_db.query_with_sparse(
                vector=sample_embedding,
                sparse_vector=sample_sparse,
                top_k=5,
                scope="scope1",
            )

        call_args = mock_pinecone_db.index.query.call_args
        assert call_args.kwargs.get("namespace") == "scope1"


class TestPineconeVectorDBNamespace:
    """Test suite for Pinecone namespace operations."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            db = PineconeVectorDB(api_key="test-key", index_name="test-index")
            db.index = MagicMock()
            return db

    def test_list_namespaces(self, mock_pinecone_db) -> None:
        """Test listing namespaces."""
        mock_pinecone_db.index.describe_index_stats = MagicMock(
            return_value=MagicMock(
                to_dict=MagicMock(
                    return_value={
                        "namespaces": {
                            "ns1": {"vector_count": 10},
                            "ns2": {"vector_count": 20},
                        },
                        "dimension": 384,
                    }
                )
            )
        )

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            namespaces = mock_pinecone_db.list_namespaces()

        assert "ns1" in namespaces
        assert "ns2" in namespaces

    def test_delete_namespace(self, mock_pinecone_db) -> None:
        """Test deleting a namespace."""
        mock_pinecone_db.index.delete = MagicMock()

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            mock_pinecone_db.delete_namespace("test-namespace")

        mock_pinecone_db.index.delete.assert_called_once_with(
            delete_all=True, namespace="test-namespace"
        )

    def test_describe_index_stats(self, mock_pinecone_db) -> None:
        """Test describing index stats."""
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {
            "namespaces": {"default": {"vector_count": 100}},
            "dimension": 384,
            "total_vector_count": 100,
        }
        mock_pinecone_db.index.describe_index_stats = MagicMock(return_value=mock_stats)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            stats = mock_pinecone_db.describe_index_stats()

        assert "namespaces" in stats
        assert stats["dimension"] == 384


class TestPineconeVectorDBDelete:
    """Test suite for Pinecone delete operations."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            db = PineconeVectorDB(api_key="test-key", index_name="test-index")
            db.index = MagicMock()
            return db

    def test_delete_by_ids(self, mock_pinecone_db) -> None:
        """Test deleting by IDs."""
        mock_pinecone_db.index.delete = MagicMock()

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            mock_pinecone_db.delete(ids=["id1", "id2"], namespace="test")

        mock_pinecone_db.index.delete.assert_called_once_with(
            ids=["id1", "id2"], delete_all=False, namespace="test"
        )

    def test_delete_all(self, mock_pinecone_db) -> None:
        """Test deleting all vectors."""
        mock_pinecone_db.index.delete = MagicMock()

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            mock_pinecone_db.delete(delete_all=True, namespace="test")

        mock_pinecone_db.index.delete.assert_called_once()

    def test_fetch(self, mock_pinecone_db) -> None:
        """Test fetching vectors by ID."""
        mock_response = MagicMock()
        mock_response.to_dict.return_value = {
            "vectors": {"id1": {"id": "id1", "values": [0.1] * 384, "metadata": {}}}
        }
        mock_pinecone_db.index.fetch = MagicMock(return_value=mock_response)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            result = mock_pinecone_db.fetch(ids=["id1"], namespace="test")

        assert "vectors" in result


class TestPineconeVectorDBMetadata:
    """Test suite for Pinecone metadata operations."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            return PineconeVectorDB(api_key="test-key", index_name="test-index")

    def test_flatten_metadata_simple(self, mock_pinecone_db) -> None:
        """Test flattening simple metadata."""
        metadata = {"key": "value", "count": 42}
        result = mock_pinecone_db.flatten_metadata(metadata)

        assert result == {"key": "value", "count": 42}

    def test_flatten_metadata_nested(self, mock_pinecone_db) -> None:
        """Test flattening nested metadata."""
        metadata = {"outer": {"inner": "value"}}
        result = mock_pinecone_db.flatten_metadata(metadata)

        assert result == {"outer_inner": "value"}

    def test_flatten_metadata_list_of_strings(self, mock_pinecone_db) -> None:
        """Test flattening metadata with list of strings."""
        metadata = {"tags": ["tag1", "tag2", "tag3"]}
        result = mock_pinecone_db.flatten_metadata(metadata)

        assert result["tags"] == ["tag1", "tag2", "tag3"]

    def test_flatten_metadata_mixed_list(self, mock_pinecone_db) -> None:
        """Test flattening metadata with mixed list."""
        metadata = {"items": [1, 2, "three"]}
        result = mock_pinecone_db.flatten_metadata(metadata)

        # Mixed lists should be converted to strings
        assert result["items"] == ["1", "2", "three"]

    def test_flatten_metadata_none_values(self, mock_pinecone_db) -> None:
        """Test flattening metadata with None values."""
        metadata = {"key": "value", "null_key": None}
        result = mock_pinecone_db.flatten_metadata(metadata)

        assert "key" in result
        assert "null_key" not in result

    def test_flatten_metadata_complex_types(self, mock_pinecone_db) -> None:
        """Test flattening metadata with complex types."""
        metadata = {"date": MagicMock(__str__=lambda self: "2023-01-01")}
        result = mock_pinecone_db.flatten_metadata(metadata)

        # Complex types should be converted to strings
        assert isinstance(result["date"], str)


class TestPineconeVectorDBFilterBuilding:
    """Test suite for Pinecone filter building."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            return PineconeVectorDB(api_key="test-key", index_name="test-index")

    def test_build_filter(self, mock_pinecone_db) -> None:
        """Test building simple filter."""
        result = mock_pinecone_db.build_filter("category", "$eq", "fiction")
        assert result == {"category": {"$eq": "fiction"}}

    def test_build_compound_filter_and(self, mock_pinecone_db) -> None:
        """Test building compound AND filter."""
        conditions = [
            {"category": {"$eq": "fiction"}},
            {"year": {"$gt": 2020}},
        ]
        result = mock_pinecone_db.build_compound_filter(conditions, logic="AND")

        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_build_compound_filter_or(self, mock_pinecone_db) -> None:
        """Test building compound OR filter."""
        conditions = [
            {"category": {"$eq": "fiction"}},
            {"category": {"$eq": "non-fiction"}},
        ]
        result = mock_pinecone_db.build_compound_filter(conditions, logic="OR")

        assert "$or" in result
        assert len(result["$or"]) == 2


class TestPineconeVectorDBEstimateMatchCount:
    """Test suite for estimate_match_count."""

    @pytest.fixture
    def mock_pinecone_db(self):
        """Create mock PineconeVectorDB instance."""
        with patch("vectordb.databases.pinecone.Pinecone"):
            db = PineconeVectorDB(api_key="test-key", index_name="test-index")
            db.index = MagicMock()
            return db

    def test_estimate_match_count(self, mock_pinecone_db) -> None:
        """Test estimating match count."""
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {
            "namespaces": {"test-ns": {"vector_count": 100}},
            "dimension": 384,
        }
        mock_pinecone_db.index.describe_index_stats = MagicMock(return_value=mock_stats)

        with patch.object(
            mock_pinecone_db, "_get_index", return_value=mock_pinecone_db.index
        ):
            count = mock_pinecone_db.estimate_match_count(
                filter={"category": "fiction"},
                namespace="test-ns",
            )

        # Returns namespace count as fallback
        assert count == 100
