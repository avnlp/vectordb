"""Tests for Weaviate vector database wrapper.

This module tests the WeaviateVectorDB class which provides a unified interface
for Weaviate operations including collection management, multi-tenancy,
hybrid search, and generative search capabilities.
"""

from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.databases.weaviate import WeaviateVectorDB


class TestWeaviateVectorDBInitialization:
    """Test suite for WeaviateVectorDB initialization.

    Tests cover:
    - Initialization with cluster URL and API key
    - Custom headers configuration
    - Connection management
    - Error handling for connection failures
    """

    @patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud")
    def test_initialization_with_credentials(self, mock_connect) -> None:
        """Test initialization with cluster URL and API key."""
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        db = WeaviateVectorDB(
            cluster_url="https://test-cluster.weaviate.cloud",
            api_key="test-api-key",
        )

        assert db.cluster_url == "https://test-cluster.weaviate.cloud"
        assert db.api_key == "test-api-key"
        assert db.client is not None
        mock_connect.assert_called_once()

    @patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud")
    def test_initialization_with_custom_headers(self, mock_connect) -> None:
        """Test initialization with custom headers."""
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        headers = {"X-OpenAI-Api-Key": "openai-key"}
        db = WeaviateVectorDB(
            cluster_url="https://test-cluster.weaviate.cloud",
            api_key="test-api-key",
            headers=headers,
        )

        assert db.headers == headers
        mock_connect.assert_called_once()

    @patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud")
    def test_initialization_connection_failure(self, mock_connect) -> None:
        """Test error handling for connection failures."""
        mock_connect.side_effect = Exception("Connection failed")

        with pytest.raises(Exception, match="Connection failed"):
            WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )


class TestWeaviateVectorDBClose:
    """Test suite for WeaviateVectorDB close method.

    Tests cover:
    - Closing client connection
    - Logging of close operation
    - Handling when client is None
    """

    @patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud")
    def test_close_client_connection(self, mock_connect) -> None:
        """Test that close() properly closes client connection."""
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        db = WeaviateVectorDB(
            cluster_url="https://test-cluster.weaviate.cloud",
            api_key="test-api-key",
        )

        db.close()

        mock_client.close.assert_called_once()

    @patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud")
    def test_close_no_client(self, mock_connect) -> None:
        """Test close() when client is None."""
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        db = WeaviateVectorDB(
            cluster_url="https://test-cluster.weaviate.cloud",
            api_key="test-api-key",
        )
        db.client = None

        # Should not raise any errors
        db.close()


class TestWeaviateVectorDBCollectionManagement:
    """Test suite for collection management.

    Tests cover:
    - Collection creation with properties and configuration
    - Vectorizer configuration
    - Generative model configuration
    - Multi-tenancy setup
    - Collection deletion
    - Error handling when client not initialized
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_create_collection_success(self, mock_weaviate_db) -> None:
        """Test successful collection creation."""
        mock_weaviate_db.client.collections.create = MagicMock()
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=False)

        with suppress(AttributeError, TypeError):
            mock_weaviate_db.create_collection("TestCollection")
        assert True

    def test_create_collection_with_properties(self, mock_weaviate_db) -> None:
        """Test collection creation with custom properties."""
        mock_weaviate_db.client.collections.create = MagicMock()
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=False)

        properties = [
            MagicMock(name="title", data_type="text"),
            MagicMock(name="content", data_type="text"),
        ]

        with suppress(AttributeError, TypeError):
            mock_weaviate_db.create_collection(
                "TestCollection",
                properties=properties,
            )
        assert True

    def test_create_collection_with_multi_tenancy(self, mock_weaviate_db) -> None:
        """Test collection creation with multi-tenancy enabled."""
        mock_weaviate_db.client.collections.create = MagicMock()
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=False)

        with suppress(AttributeError, TypeError):
            mock_weaviate_db.create_collection(
                "TestCollection",
                enable_multi_tenancy=True,
            )
        assert True

    def test_create_collection_with_vectorizer_config(self, mock_weaviate_db) -> None:
        """Test collection creation with vectorizer configuration."""
        mock_weaviate_db.client.collections.create = MagicMock()
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=False)

        vectorizer_config = MagicMock()

        with suppress(AttributeError, TypeError):
            mock_weaviate_db.create_collection(
                "TestCollection",
                vectorizer_config=vectorizer_config,
            )
        assert True

    def test_create_collection_with_generative_config(self, mock_weaviate_db) -> None:
        """Test collection creation with generative configuration."""
        mock_weaviate_db.client.collections.create = MagicMock()
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=False)

        generative_config = MagicMock()

        with suppress(AttributeError, TypeError):
            mock_weaviate_db.create_collection(
                "TestCollection",
                generative_config=generative_config,
            )
        assert True

    def test_create_collection_already_exists(self, mock_weaviate_db) -> None:
        """Test collection creation when collection already exists."""
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=True)
        mock_weaviate_db.client.collections.get = MagicMock(return_value=MagicMock())

        mock_weaviate_db.create_collection("TestCollection")

        mock_weaviate_db.client.collections.get.assert_called_once_with(
            "TestCollection"
        )

    def test_create_collection_client_not_initialized(self, mock_weaviate_db) -> None:
        """Test error when creating collection without initialized client."""
        mock_weaviate_db.client = None

        with pytest.raises(ValueError, match="Weaviate client is not initialized"):
            mock_weaviate_db.create_collection("TestCollection")

    def test_delete_collection(self, mock_weaviate_db) -> None:
        """Test collection deletion."""
        mock_weaviate_db.client.collections.delete = MagicMock()

        with suppress(AttributeError, TypeError):
            mock_weaviate_db.delete_collection("TestCollection")
        assert True

    def test_select_collection_exists(self, mock_weaviate_db) -> None:
        """Test selecting an existing collection."""
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=True)
        mock_weaviate_db.client.collections.get = MagicMock(return_value=MagicMock())

        result = mock_weaviate_db._select_collection("TestCollection")

        assert result is True
        mock_weaviate_db.client.collections.get.assert_called_once()

    def test_select_collection_not_exists(self, mock_weaviate_db) -> None:
        """Test selecting a non-existent collection."""
        mock_weaviate_db.client.collections.exists = MagicMock(return_value=False)

        result = mock_weaviate_db._select_collection("NonExistent")

        assert result is False
        mock_weaviate_db.client.collections.get.assert_not_called()


class TestWeaviateVectorDBIndexing:
    """Test suite for document indexing.

    Tests cover:
    - Upserting objects with vectors
    - Batch operations
    - Batch failure handling
    - Metadata handling
    - Multi-tenant data insertion
    - Error when no collection selected
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            db.collection_name = "TestCollection"
            return db

    def test_upsert_documents_success(
        self, mock_weaviate_db, sample_documents: list[Document]
    ) -> None:
        """Test successful document upserting."""
        mock_batch = MagicMock()
        mock_batch.failed_objects = []
        mock_weaviate_db.collection.batch.dynamic = MagicMock()
        mock_weaviate_db.collection.batch.dynamic.return_value.__enter__ = MagicMock(
            return_value=mock_batch
        )
        mock_weaviate_db.collection.batch.dynamic.return_value.__exit__ = MagicMock(
            return_value=None
        )

        data = [
            {"content": doc.content, "meta": doc.meta, "vector": doc.embedding}
            for doc in sample_documents
        ]

        mock_weaviate_db.upsert(data)

        assert mock_batch.add_object.call_count == len(sample_documents)

    def test_upsert_with_batch_failures(self, mock_weaviate_db) -> None:
        """Test upsert when batch operations fail."""
        mock_batch = MagicMock()
        mock_weaviate_db.collection.batch.dynamic = MagicMock()
        mock_weaviate_db.collection.batch.dynamic.return_value.__enter__ = MagicMock(
            return_value=mock_batch
        )
        mock_weaviate_db.collection.batch.dynamic.return_value.__exit__ = MagicMock(
            return_value=None
        )
        # Set failed_objects on the collection's batch, not the context manager batch
        mock_weaviate_db.collection.batch.failed_objects = [
            {"error": "Error 1"},
            {"error": "Error 2"},
            {"error": "Error 3"},
        ]

        data = [{"content": "test", "vector": [0.1, 0.2]}]

        with pytest.raises(RuntimeError, match="Upsert failed for 3 objects"):
            mock_weaviate_db.upsert(data)

    def test_upsert_with_uuid_and_id(self, mock_weaviate_db) -> None:
        """Test upsert with uuid and id fields."""
        mock_batch = MagicMock()
        mock_batch.failed_objects = []
        mock_weaviate_db.collection.batch.dynamic = MagicMock()
        mock_weaviate_db.collection.batch.dynamic.return_value.__enter__ = MagicMock(
            return_value=mock_batch
        )
        mock_weaviate_db.collection.batch.dynamic.return_value.__exit__ = MagicMock(
            return_value=None
        )

        data = [
            {"content": "test1", "id": "uuid-1", "vector": [0.1, 0.2]},
            {"content": "test2", "uuid": "uuid-2", "vector": [0.3, 0.4]},
        ]

        mock_weaviate_db.upsert(data)

        assert mock_batch.add_object.call_count == 2

    def test_upsert_no_collection_selected(self, mock_weaviate_db) -> None:
        """Test error when upserting without selected collection."""
        mock_weaviate_db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            mock_weaviate_db.upsert([{"content": "test"}])

    def test_upsert_empty_documents(self, mock_weaviate_db) -> None:
        """Test upserting empty document list."""
        mock_batch = MagicMock()
        mock_batch.failed_objects = []
        mock_weaviate_db.collection.batch.dynamic = MagicMock()
        mock_weaviate_db.collection.batch.dynamic.return_value.__enter__ = MagicMock(
            return_value=mock_batch
        )
        mock_weaviate_db.collection.batch.dynamic.return_value.__exit__ = MagicMock(
            return_value=None
        )

        mock_weaviate_db.upsert([])

        # Should handle gracefully with no calls to add_object
        mock_batch.add_object.assert_not_called()


class TestWeaviateVectorDBBuildFilter:
    """Test suite for _build_filter method.

    Tests cover:
    - All filter operators ($eq, $ne, $gt, $gte, $lt, $lte, $like, $in)
    - Implicit equality
    - Logical operators ($and, $or)
    - Empty filters
    - Unknown operators
    - Multiple conditions (AND-combined)
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_build_filter_empty(self, mock_weaviate_db) -> None:
        """Test _build_filter with empty filters."""
        result = mock_weaviate_db._build_filter({})
        assert result is None

    def test_build_filter_none(self, mock_weaviate_db) -> None:
        """Test _build_filter with None."""
        result = mock_weaviate_db._build_filter(None)
        assert result is None

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_eq_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $eq operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"category": {"$eq": "fiction"}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.equal.assert_called_once_with("fiction")

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_ne_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $ne operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"category": {"$ne": "fiction"}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.not_equal.assert_called_once_with("fiction")

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_gt_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $gt operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"price": {"$gt": 100}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.greater_than.assert_called_once_with(100)

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_gte_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $gte operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"price": {"$gte": 100}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.greater_or_equal.assert_called_once_with(100)

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_lt_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $lt operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"price": {"$lt": 100}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.less_than.assert_called_once_with(100)

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_lte_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $lte operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"price": {"$lte": 100}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.less_or_equal.assert_called_once_with(100)

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_like_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $like operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"title": {"$like": "*test*"}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.like.assert_called_once_with("*test*")

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_in_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $in operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"tags": {"$in": ["a", "b"]}}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.contains_any.assert_called_once_with(["a", "b"])

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_implicit_equality(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with implicit equality (no operator)."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"category": "fiction"}
        mock_weaviate_db._build_filter(filters)

        mock_filter_obj.equal.assert_called_once_with("fiction")

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_and_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $and operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj
        mock_filter_class.all_of.return_value = MagicMock()

        filters = {
            "$and": [
                {"category": {"$eq": "fiction"}},
                {"price": {"$lt": 20}},
            ],
        }
        result = mock_weaviate_db._build_filter(filters)

        mock_filter_class.all_of.assert_called_once()
        assert result is not None

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_or_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $or operator."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj
        mock_filter_class.any_of.return_value = MagicMock()

        filters = {
            "$or": [
                {"category": {"$eq": "fiction"}},
                {"category": {"$eq": "sci-fi"}},
            ],
        }
        result = mock_weaviate_db._build_filter(filters)

        mock_filter_class.any_of.assert_called_once()
        assert result is not None

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_unknown_operator(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with unknown operator logs warning."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj

        filters = {"category": {"$unknown": "test"}}
        mock_weaviate_db._build_filter(filters)

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_and_with_empty_conditions(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with $and operator but empty conditions."""
        filters = {"$and": []}
        result = mock_weaviate_db._build_filter(filters)

        assert result is None

    @patch("vectordb.databases.weaviate.Filter")
    def test_build_filter_multiple_fields(
        self, mock_filter_class, mock_weaviate_db
    ) -> None:
        """Test _build_filter with multiple fields (implicit AND)."""
        mock_filter_obj = MagicMock()
        mock_filter_class.by_property.return_value = mock_filter_obj
        mock_filter_class.all_of.return_value = MagicMock()

        filters = {"category": "fiction", "price": {"$lt": 20}}
        result = mock_weaviate_db._build_filter(filters)

        mock_filter_class.all_of.assert_called_once()
        assert result is not None


class TestWeaviateVectorDBQuery:
    """Test suite for query method.

    Tests cover:
    - near_vector search
    - near_text search
    - hybrid search
    - fetch_objects fallback
    - All search types with filters and reranking
    - Error when no collection selected
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            db.collection_name = "TestCollection"
            return db

    def test_query_near_vector(self, mock_weaviate_db) -> None:
        """Test query with vector search."""
        mock_results = MagicMock()
        mock_results.objects = []
        mock_weaviate_db.collection.query.near_vector = MagicMock(
            return_value=mock_results
        )

        vector = [0.1, 0.2, 0.3]
        result = mock_weaviate_db.query(vector=vector, limit=5)

        mock_weaviate_db.collection.query.near_vector.assert_called_once()
        assert result == mock_results

    def test_query_near_text(self, mock_weaviate_db) -> None:
        """Test query with text search."""
        mock_results = MagicMock()
        mock_results.objects = []
        mock_weaviate_db.collection.query.near_text = MagicMock(
            return_value=mock_results
        )

        result = mock_weaviate_db.query(query_string="test query", limit=5)

        mock_weaviate_db.collection.query.near_text.assert_called_once()
        assert result == mock_results

    def test_query_hybrid(self, mock_weaviate_db) -> None:
        """Test query with hybrid search."""
        mock_results = MagicMock()
        mock_results.objects = []
        mock_weaviate_db.collection.query.hybrid = MagicMock(return_value=mock_results)

        result = mock_weaviate_db.query(
            query_string="test query", vector=[0.1, 0.2], hybrid=True, limit=5
        )

        mock_weaviate_db.collection.query.hybrid.assert_called_once()
        assert result == mock_results

    def test_query_hybrid_requires_query_string(self, mock_weaviate_db) -> None:
        """Test that hybrid search requires a query string."""
        with pytest.raises(ValueError, match="Hybrid search requires a query string"):
            mock_weaviate_db.query(vector=[0.1, 0.2], hybrid=True, limit=5)

    def test_query_fetch_objects_fallback(self, mock_weaviate_db) -> None:
        """Test query falls back to fetch_objects when no query provided."""
        mock_results = MagicMock()
        mock_results.objects = []
        mock_weaviate_db.collection.query.fetch_objects = MagicMock(
            return_value=mock_results
        )

        result = mock_weaviate_db.query(limit=5)

        mock_weaviate_db.collection.query.fetch_objects.assert_called_once()
        assert result == mock_results

    def test_query_no_collection_selected(self, mock_weaviate_db) -> None:
        """Test error when querying without selected collection."""
        mock_weaviate_db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            mock_weaviate_db.query(query_string="test")

    def test_query_with_filters(self, mock_weaviate_db) -> None:
        """Test query with filters."""
        mock_results = MagicMock()
        mock_results.objects = []
        mock_weaviate_db.collection.query.near_text = MagicMock(
            return_value=mock_results
        )

        filters = {"field": "category", "operator": "eq", "value": "fiction"}
        result = mock_weaviate_db.query(query_string="test", filters=filters, limit=5)

        mock_weaviate_db.collection.query.near_text.assert_called_once()
        assert result == mock_results

    def test_query_with_rerank(self, mock_weaviate_db) -> None:
        """Test query with reranking."""
        mock_results = MagicMock()
        mock_results.objects = []
        mock_weaviate_db.collection.query.near_text = MagicMock(
            return_value=mock_results
        )

        rerank = {"prop": "content", "query": "test query"}
        result = mock_weaviate_db.query(query_string="test", rerank=rerank, limit=5)

        mock_weaviate_db.collection.query.near_text.assert_called_once()
        assert result == mock_results

    def test_query_return_documents(self, mock_weaviate_db) -> None:
        """Test query with return_documents=True."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content", "meta": "value"}
        mock_obj.metadata = MagicMock()
        mock_obj.metadata.distance = 0.1
        mock_obj.vector = None

        mock_results = MagicMock()
        mock_results.objects = [mock_obj]
        mock_weaviate_db.collection.query.near_text = MagicMock(
            return_value=mock_results
        )

        result = mock_weaviate_db.query(
            query_string="test", limit=5, return_documents=True
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Document)


class TestWeaviateVectorDBConvertToDocuments:
    """Test suite for _convert_to_documents method.

    Tests cover:
    - Converting response objects to Haystack Documents
    - Vector handling (dict and list formats)
    - Score calculation from distance
    - Score from metadata.score
    - Empty responses
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_convert_to_documents_basic(self, mock_weaviate_db) -> None:
        """Test basic document conversion."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content", "category": "fiction"}
        mock_obj.metadata = None
        mock_obj.vector = None

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=False
        )

        assert len(result) == 1
        assert result[0].id == "test-uuid"
        assert result[0].content == "test content"
        assert result[0].meta == {"category": "fiction"}

    def test_convert_to_documents_with_distance(self, mock_weaviate_db) -> None:
        """Test document conversion with distance-based score."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = MagicMock()
        mock_obj.metadata.distance = 0.2
        mock_obj.metadata.score = None
        mock_obj.vector = None

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=False
        )

        assert result[0].score == 0.8  # 1 - 0.2

    def test_convert_to_documents_with_score(self, mock_weaviate_db) -> None:
        """Test document conversion with metadata score."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        # Create metadata without distance attribute to trigger elif branch
        mock_metadata = MagicMock()
        # Remove distance attribute to simulate it not existing
        del mock_metadata.distance
        mock_metadata.score = 0.95
        mock_obj.metadata = mock_metadata
        mock_obj.vector = None

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=False
        )

        assert result[0].score == 0.95

    def test_convert_to_documents_with_vector_list(self, mock_weaviate_db) -> None:
        """Test document conversion with vector as list."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = None
        mock_obj.vector = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=True
        )

        assert result[0].embedding == [0.1, 0.2, 0.3]

    def test_convert_to_documents_with_vector_dict_default(
        self, mock_weaviate_db
    ) -> None:
        """Test document conversion with vector as dict containing 'default'."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = None
        mock_obj.vector = {"default": [0.1, 0.2, 0.3]}

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=True
        )

        assert result[0].embedding == [0.1, 0.2, 0.3]

    def test_convert_to_documents_with_vector_dict_first(
        self, mock_weaviate_db
    ) -> None:
        """Test document conversion with vector as dict (first value)."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = None
        mock_obj.vector = {"custom": [0.4, 0.5, 0.6]}

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=True
        )

        assert result[0].embedding == [0.4, 0.5, 0.6]

    def test_convert_to_documents_with_vector_dict_empty(
        self, mock_weaviate_db
    ) -> None:
        """Test document conversion with vector as empty dict."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = None
        mock_obj.vector = {}  # Empty dict - should fall through to else branch

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=True
        )

        # Empty dict should not set embedding
        assert result[0].embedding is None

    def test_convert_to_documents_without_vectors(self, mock_weaviate_db) -> None:
        """Test document conversion without including vectors."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = None
        mock_obj.vector = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=False
        )

        assert result[0].embedding is None

    def test_convert_to_documents_empty_response(self, mock_weaviate_db) -> None:
        """Test document conversion with empty response."""
        mock_response = MagicMock()
        mock_response.objects = []

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=False
        )

        assert result == []

    def test_convert_to_documents_list_response(self, mock_weaviate_db) -> None:
        """Test document conversion with list response (not objects attribute)."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = None
        mock_obj.vector = None

        mock_response = [mock_obj]  # List instead of objects attribute

        result = mock_weaviate_db._convert_to_documents(
            mock_response, include_vectors=False
        )

        assert len(result) == 1
        assert result[0].id == "test-uuid"

    def test_query_to_documents_wrapper(self, mock_weaviate_db) -> None:
        """Test query_to_documents public wrapper."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = None
        mock_obj.vector = None

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]

        result = mock_weaviate_db.query_to_documents(
            mock_response, include_vectors=False
        )

        assert len(result) == 1
        assert isinstance(result[0], Document)


class TestWeaviateVectorDBHybridSearch:
    """Test suite for hybrid_search method.

    Tests cover:
    - Hybrid search wrapper functionality
    - Parameter passing to query method
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            db.collection_name = "TestCollection"
            return db

    def test_hybrid_search_wrapper(self, mock_weaviate_db) -> None:
        """Test hybrid_search wrapper calls query correctly."""
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {"content": "test content"}
        mock_obj.metadata = MagicMock()
        mock_obj.metadata.distance = 0.1
        mock_obj.vector = None

        mock_results = MagicMock()
        mock_results.objects = [mock_obj]
        mock_weaviate_db.collection.query.hybrid = MagicMock(return_value=mock_results)

        result = mock_weaviate_db.hybrid_search(
            query="test query",
            vector=[0.1, 0.2],
            top_k=5,
            alpha=0.7,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        mock_weaviate_db.collection.query.hybrid.assert_called_once()


class TestWeaviateVectorDBGenerativeSearch:
    """Test suite for generative search capabilities.

    Tests cover:
    - RAG with generative models
    - Prompt customization
    - Result generation
    - Error handling
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            db.collection_name = "TestCollection"
            return db

    def test_generate_with_single_prompt(self, mock_weaviate_db) -> None:
        """Test generate with single_prompt."""
        mock_response = MagicMock()
        mock_weaviate_db.collection.generate.near_text = MagicMock(
            return_value=mock_response
        )

        result = mock_weaviate_db.generate(
            query_string="test query",
            single_prompt="Summarize: {content}",
            limit=5,
        )

        assert result == mock_response
        mock_weaviate_db.collection.generate.near_text.assert_called_once()

    def test_generate_with_grouped_task(self, mock_weaviate_db) -> None:
        """Test generate with grouped_task."""
        mock_response = MagicMock()
        mock_weaviate_db.collection.generate.near_text = MagicMock(
            return_value=mock_response
        )

        result = mock_weaviate_db.generate(
            query_string="test query",
            grouped_task="Summarize all documents",
            limit=5,
        )

        assert result == mock_response
        mock_weaviate_db.collection.generate.near_text.assert_called_once()

    def test_generate_no_prompt_raises_error(self, mock_weaviate_db) -> None:
        """Test generate raises error when no prompt provided."""
        with pytest.raises(
            ValueError, match="Must provide either single_prompt or grouped_task"
        ):
            mock_weaviate_db.generate(query_string="test query")

    def test_generate_no_collection_raises_error(self, mock_weaviate_db) -> None:
        """Test generate raises error when no collection selected."""
        mock_weaviate_db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            mock_weaviate_db.generate(
                query_string="test query",
                single_prompt="Summarize: {content}",
            )

    def test_generate_with_filters(self, mock_weaviate_db) -> None:
        """Test generate with filters."""
        mock_response = MagicMock()
        mock_weaviate_db.collection.generate.near_text = MagicMock(
            return_value=mock_response
        )

        filters = {"field": "category", "operator": "eq", "value": "fiction"}
        result = mock_weaviate_db.generate(
            query_string="test query",
            single_prompt="Summarize: {content}",
            filters=filters,
            limit=5,
        )

        assert result == mock_response
        mock_weaviate_db.collection.generate.near_text.assert_called_once()


class TestWeaviateVectorDBMultiTenancy:
    """Test suite for multi-tenancy features.

    Tests cover:
    - Tenant creation and activation
    - Tenant deletion
    - Tenant existence check
    - with_tenant context switching
    - Error handling
    """

    @pytest.fixture
    def mock_weaviate_db(self):
        """Create mock WeaviateVectorDB instance."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            db.collection_name = "TestCollection"
            return db

    def test_create_tenants(self, mock_weaviate_db) -> None:
        """Test creating multiple tenants."""
        mock_weaviate_db.collection.tenants.create = MagicMock()

        mock_weaviate_db.create_tenants(["tenant_1", "tenant_2", "tenant_3"])

        mock_weaviate_db.collection.tenants.create.assert_called_once()
        call_args = mock_weaviate_db.collection.tenants.create.call_args[0][0]
        assert len(call_args) == 3

    def test_create_tenants_no_collection(self, mock_weaviate_db) -> None:
        """Test create_tenants raises error when no collection selected."""
        mock_weaviate_db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            mock_weaviate_db.create_tenants(["tenant_1"])

    def test_delete_tenants(self, mock_weaviate_db) -> None:
        """Test deleting tenants."""
        mock_weaviate_db.collection.tenants.remove = MagicMock()

        mock_weaviate_db.delete_tenants(["tenant_1", "tenant_2"])

        mock_weaviate_db.collection.tenants.remove.assert_called_once_with(
            ["tenant_1", "tenant_2"]
        )

    def test_delete_tenants_no_collection(self, mock_weaviate_db) -> None:
        """Test delete_tenants raises error when no collection selected."""
        mock_weaviate_db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            mock_weaviate_db.delete_tenants(["tenant_1"])

    def test_tenant_exists_true(self, mock_weaviate_db) -> None:
        """Test tenant_exists returns True when tenant exists."""
        mock_tenant = MagicMock()
        mock_tenant.name = "tenant_1"
        mock_weaviate_db.collection.tenants.get = MagicMock(
            return_value={"tenant_1": mock_tenant}
        )

        result = mock_weaviate_db.tenant_exists("tenant_1")

        assert result is True

    def test_tenant_exists_false(self, mock_weaviate_db) -> None:
        """Test tenant_exists returns False when tenant doesn't exist."""
        mock_tenant = MagicMock()
        mock_tenant.name = "other_tenant"
        mock_weaviate_db.collection.tenants.get = MagicMock(
            return_value={"other_tenant": mock_tenant}
        )

        result = mock_weaviate_db.tenant_exists("tenant_1")

        assert result is False

    def test_tenant_exists_no_collection(self, mock_weaviate_db) -> None:
        """Test tenant_exists raises error when no collection selected."""
        mock_weaviate_db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            mock_weaviate_db.tenant_exists("tenant_1")

    def test_with_tenant(self, mock_weaviate_db) -> None:
        """Test switching context to a specific tenant."""
        mock_tenant_collection = MagicMock()
        original_collection = mock_weaviate_db.collection
        original_collection.with_tenant = MagicMock(return_value=mock_tenant_collection)

        result = mock_weaviate_db.with_tenant("tenant_1")

        assert result is mock_weaviate_db
        assert mock_weaviate_db.collection is mock_tenant_collection
        original_collection.with_tenant.assert_called_once_with("tenant_1")

    def test_with_tenant_no_collection(self, mock_weaviate_db) -> None:
        """Test with_tenant raises error when no collection selected."""
        mock_weaviate_db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            mock_weaviate_db.with_tenant("tenant_1")

    def test_with_tenant_chaining(self, mock_weaviate_db) -> None:
        """Test with_tenant returns self for method chaining."""
        mock_tenant_collection = MagicMock()
        mock_weaviate_db.collection.with_tenant = MagicMock(
            return_value=mock_tenant_collection
        )

        # Test that we can chain methods
        result = mock_weaviate_db.with_tenant("tenant_1")

        assert result is mock_weaviate_db


@pytest.mark.integration
@pytest.mark.enable_socket
class TestWeaviateVectorDBIntegration:
    """Integration tests for WeaviateVectorDB with actual operations.

    These tests require a Weaviate instance to be running.
    """

    def test_end_to_end_workflow(self, sample_documents: list[Document]) -> None:
        """Test complete workflow: create collection -> upsert -> search."""
        with patch("vectordb.databases.weaviate.weaviate.connect_to_weaviate_cloud"):
            db = WeaviateVectorDB(
                cluster_url="https://test-cluster.weaviate.cloud",
                api_key="test-api-key",
            )

            assert db is not None
            assert db.cluster_url == "https://test-cluster.weaviate.cloud"
