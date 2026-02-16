"""Tests for the WeaviateVectorDB class."""

from unittest import mock

import pytest

from vectordb.weaviate import WeaviateVectorDB


class TestWeaviateVectorDB:
    """Test cases for WeaviateVectorDB class."""

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_initialization_with_proper_parameters(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test initialization with proper parameters."""
        mock_client = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = False

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
            headers={"X-Custom-Header": "value"},
            collection_name="test_collection",
            tracing_project_name="test_project",
            weave_params={"custom_param": "value"},
        )

        assert db.cluster_url == "https://test.weaviate.cloud"
        assert db.api_key == "test-api-key"
        assert db.headers == {"X-Custom-Header": "value"}
        assert db.collection_name == "test_collection"
        assert db.tracing_project_name == "test_project"
        assert db.weave_params == {"custom_param": "value"}
        assert db.client is not None

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_initialize_client_calls_connect_to_weaviate_cloud(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test _initialize_client calls connect_to_weaviate_cloud."""
        mock_client = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = False

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
        )

        mock_weaviate_module.connect_to_weaviate_cloud.assert_called_once_with(
            cluster_url="https://test.weaviate.cloud",
            auth_credentials=mock_auth.api_key.return_value,
            headers={},
        )
        assert db.client == mock_client

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_select_collection_returns_true_if_collection_exists(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test _select_collection returns True if collection exists."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_collection

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
        )

        result = db._select_collection("existing_collection")

        assert result is True
        assert db.collection == mock_collection
        mock_client.collections.get.assert_called_once_with("existing_collection")

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_select_collection_returns_false_if_collection_does_not_exist(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test _select_collection returns False if collection does not exist."""
        mock_client = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = False

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
        )

        result = db._select_collection("nonexistent_collection")

        assert result is False
        assert db.collection is None

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    @mock.patch("vectordb.weaviate.Configure")
    def test_create_collection_uses_existing_collection_if_present(
        self, mock_configure, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test create_collection uses existing collection if present."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_collection

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
        )

        db.create_collection("existing_collection")

        mock_client.collections.create.assert_not_called()
        assert db.collection == mock_collection
        assert db.collection_name == "existing_collection"

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    @mock.patch("vectordb.weaviate.Configure")
    @mock.patch("vectordb.weaviate.Property")
    def test_create_collection_creates_new_collection_if_not_exists(
        self, mock_property, mock_configure, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test create_collection creates new collection if not exists."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = False

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
        )

        mock_properties = [mock.MagicMock()]
        mock_vectorizer_config = mock.MagicMock()
        mock_generative_config = mock.MagicMock()

        mock_client.collections.create.return_value = None
        mock_client.collections.get.return_value = mock_collection

        db.create_collection(
            "new_collection",
            properties=mock_properties,
            vectorizer_config=mock_vectorizer_config,
            generative_config=mock_generative_config,
        )

        mock_client.collections.create.assert_called_once_with(
            name="new_collection",
            properties=mock_properties,
            vectorizer_config=mock_vectorizer_config,
            generative_config=mock_generative_config,
        )
        assert db.collection_name == "new_collection"

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_create_collection_raises_error_if_client_not_initialized(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test create_collection raises ValueError if client not initialized."""
        mock_client = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = False

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
        )

        db.client = None

        with pytest.raises(ValueError, match="Weaviate client is not initialized."):
            db.create_collection("test_collection")

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_upsert_uses_batch_dynamic_context_manager(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test upsert uses batch dynamic context manager."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_batch = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_collection
        mock_collection.batch.dynamic.return_value.__enter__ = mock.MagicMock(
            return_value=mock_batch
        )
        mock_collection.batch.dynamic.return_value.__exit__ = mock.MagicMock()
        mock_collection.batch.failed_objects = []

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
            collection_name="test_collection",
        )

        data = [
            {"vector": [0.1, 0.2], "uuid": "uuid-1", "name": "item1"},
            {"vector": [0.3, 0.4], "uuid": "uuid-2", "name": "item2"},
        ]

        db.upsert(data)

        mock_collection.batch.dynamic.assert_called_once()
        assert mock_batch.add_object.call_count == 2
        mock_batch.add_object.assert_any_call(
            properties={"name": "item1"}, vector=[0.1, 0.2], uuid="uuid-1"
        )
        mock_batch.add_object.assert_any_call(
            properties={"name": "item2"}, vector=[0.3, 0.4], uuid="uuid-2"
        )

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_upsert_raises_runtime_error_on_failure(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test upsert raises RuntimeError when objects fail."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_batch = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_collection
        mock_collection.batch.dynamic.return_value.__enter__ = mock.MagicMock(
            return_value=mock_batch
        )
        mock_collection.batch.dynamic.return_value.__exit__ = mock.MagicMock()
        mock_collection.batch.failed_objects = [{"error": "failed"}]

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
            collection_name="test_collection",
        )

        data = [{"vector": [0.1, 0.2], "uuid": "uuid-1", "name": "item1"}]

        with pytest.raises(RuntimeError, match="Upsert failed for 1 objects."):
            db.upsert(data)

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    @mock.patch("vectordb.weaviate.MetadataQuery")
    def test_query_with_near_vector(
        self, mock_metadata_query, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test query with near_vector (non-hybrid)."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_query_result = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = mock_query_result

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
            collection_name="test_collection",
        )

        query_vector = [0.1, 0.2, 0.3]
        result = db.query(vector=query_vector, limit=5)

        mock_collection.query.near_vector.assert_called_once_with(
            vector=query_vector,
            limit=5,
            filters=None,
            return_metadata=mock_metadata_query.return_value,
        )
        assert result == mock_query_result

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    @mock.patch("vectordb.weaviate.MetadataQuery")
    def test_query_with_hybrid_search(
        self, mock_metadata_query, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test query with hybrid search."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_query_result = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.hybrid.return_value = mock_query_result

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
            collection_name="test_collection",
        )

        query_vector = [0.1, 0.2, 0.3]
        result = db.query(
            vector=query_vector,
            limit=10,
            hybrid=True,
            alpha=0.7,
            query_string="search query",
        )

        mock_collection.query.hybrid.assert_called_once_with(
            query="search query",
            alpha=0.7,
            vector=query_vector,
            limit=10,
            filters=None,
            return_metadata=mock_metadata_query.return_value,
        )
        assert result == mock_query_result

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    @mock.patch("vectordb.weaviate.MetadataQuery")
    def test_query_with_filters(
        self, mock_metadata_query, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test query with filters."""
        mock_client = mock.MagicMock()
        mock_collection = mock.MagicMock()
        mock_query_result = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = True
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.return_value = mock_query_result

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
            collection_name="test_collection",
        )

        mock_filter = mock.MagicMock()
        query_vector = [0.1, 0.2, 0.3]
        result = db.query(vector=query_vector, limit=5, filters=mock_filter)

        mock_collection.query.near_vector.assert_called_once_with(
            vector=query_vector,
            limit=5,
            filters=mock_filter,
            return_metadata=mock_metadata_query.return_value,
        )
        assert result == mock_query_result

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_initialization_with_default_headers(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test initialization with default empty headers."""
        mock_client = mock.MagicMock()
        mock_weaviate_module.connect_to_weaviate_cloud.return_value = mock_client
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        mock_client.collections.exists.return_value = False

        db = WeaviateVectorDB(
            cluster_url="https://test.weaviate.cloud",
            api_key="test-api-key",
        )

        assert db.headers == {}
        mock_weaviate_module.connect_to_weaviate_cloud.assert_called_once_with(
            cluster_url="https://test.weaviate.cloud",
            auth_credentials=mock_auth.api_key.return_value,
            headers={},
        )

    @mock.patch("vectordb.weaviate.weaviate")
    @mock.patch("vectordb.weaviate.Auth")
    @mock.patch("vectordb.weaviate.weave")
    def test_initialization_failure_raises_exception(
        self, mock_weave, mock_auth, mock_weaviate_module
    ):
        """Test initialization failure raises exception."""
        mock_weaviate_module.connect_to_weaviate_cloud.side_effect = Exception(
            "Connection failed"
        )
        mock_auth.api_key.return_value = "auth_credentials"
        mock_weave.init.return_value = None

        with pytest.raises(Exception, match="Connection failed"):
            WeaviateVectorDB(
                cluster_url="https://test.weaviate.cloud",
                api_key="test-api-key",
            )
