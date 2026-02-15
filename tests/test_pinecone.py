"""Tests for the Pinecone VectorDB module."""

from unittest import mock

import pytest

from vectordb.pinecone import PineconeVectorDB


class TestPineconeVectorDB:
    """Test cases for PineconeVectorDB class."""

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_init_creates_client_properly(self, mock_weave_init, mock_pinecone_class):
        """Test that initialization creates Pinecone client with correct parameters."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(
            api_key="test-api-key",
            host="test-host",
            proxy_url="http://proxy.example.com",
            proxy_headers={"Authorization": "Bearer token"},
            ssl_ca_certs="/path/to/certs",
            ssl_verify=False,
            additional_headers={"X-Custom": "value"},
            pool_threads=5,
            index_name="test-index",
            tracing_project_name="test-project",
            weave_params={"entity": "test-entity"},
        )

        mock_pinecone_class.assert_called_once_with(
            api_key="test-api-key",
            environment="test-host",
            proxy="http://proxy.example.com",
            proxy_headers={"Authorization": "Bearer token"},
            ssl_ca_certs="/path/to/certs",
            verify_ssl=False,
            additional_headers={"X-Custom": "value"},
            pool_threads=5,
        )
        assert db.client is mock_client
        assert db.api_key == "test-api-key"
        assert db.host == "test-host"
        mock_weave_init.assert_called_once_with("test-project", entity="test-entity")

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_init_without_index_name(self, mock_weave_init, mock_pinecone_class):
        """Test initialization without an index name."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")

        assert db.index_name is None
        assert db.index is None

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_select_index_returns_true_when_index_exists(
        self, mock_weave_init, mock_pinecone_class
    ):
        """Test _select_index returns True when index exists."""
        mock_client = mock.MagicMock()
        mock_client.list_indexes.return_value = [
            {"name": "existing-index"},
            {"name": "another-index"},
        ]
        mock_index = mock.MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        result = db._select_index("existing-index")

        assert result is True
        assert db.index is mock_index
        mock_client.Index.assert_called_once_with("existing-index")

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_select_index_returns_false_when_index_not_exists(
        self, mock_weave_init, mock_pinecone_class
    ):
        """Test _select_index returns False when index does not exist."""
        mock_client = mock.MagicMock()
        mock_client.list_indexes.return_value = [{"name": "other-index"}]
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        result = db._select_index("non-existent-index")

        assert result is False
        assert db.index is None

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_create_index_selects_existing_index(
        self, mock_weave_init, mock_pinecone_class
    ):
        """Test create_index selects existing index if it exists."""
        mock_client = mock.MagicMock()
        mock_client.list_indexes.return_value = [{"name": "existing-index"}]
        mock_index = mock.MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        db.create_index("existing-index", dimension=1536)

        assert db.index_name == "existing-index"
        assert db.index is mock_index
        mock_client.create_index.assert_not_called()

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    @mock.patch("vectordb.pinecone.ServerlessSpec")
    def test_create_index_creates_new_index_and_waits_for_ready(
        self, mock_spec_class, mock_weave_init, mock_pinecone_class
    ):
        """Test create_index creates new index and waits for it to be ready."""
        mock_client = mock.MagicMock()
        # First call returns empty (index doesn't exist), second call returns the index
        mock_client.list_indexes.side_effect = [[], [{"name": "new-index"}]]
        mock_client.describe_index.return_value.status = {"ready": True}
        mock_index = mock.MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_spec = mock.MagicMock()
        mock_spec_class.return_value = mock_spec

        db = PineconeVectorDB(api_key="test-api-key")
        db.create_index("new-index", dimension=1536, metric="euclidean")

        mock_client.create_index.assert_called_once_with(
            name="new-index",
            dimension=1536,
            metric="euclidean",
            spec=mock_spec,
            deletion_protection="disabled",
        )
        mock_client.describe_index.assert_called_with("new-index")
        assert db.index is mock_index

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_create_index_uses_default_spec(self, mock_weave_init, mock_pinecone_class):
        """Test create_index uses default ServerlessSpec when none provided."""
        mock_client = mock.MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.describe_index.return_value.status = {"ready": True}
        mock_client.Index.return_value = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        with mock.patch("vectordb.pinecone.ServerlessSpec") as mock_spec_class:
            mock_spec = mock.MagicMock()
            mock_spec_class.return_value = mock_spec

            db = PineconeVectorDB(api_key="test-api-key")
            db.create_index("test-index", dimension=768)

            mock_spec_class.assert_called_once_with(cloud="aws", region="us-east-1")
            mock_client.create_index.assert_called_once_with(
                name="test-index",
                dimension=768,
                metric="cosine",
                spec=mock_spec,
                deletion_protection="disabled",
            )

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_upsert_raises_valueerror_when_no_index_selected(
        self, mock_weave_init, mock_pinecone_class
    ):
        """Test upsert raises ValueError when no index is selected."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        db.index = None

        with pytest.raises(ValueError, match="No index selected"):
            db.upsert([{"id": "1", "values": [0.1, 0.2]}])

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_upsert_with_index_selected(self, mock_weave_init, mock_pinecone_class):
        """Test upsert works when index is selected."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        mock_index = mock.MagicMock()
        db.index = mock_index
        db.index_name = "test-index"

        data = [{"id": "1", "values": [0.1, 0.2]}, {"id": "2", "values": [0.3, 0.4]}]
        db.upsert(data, namespace="test-ns", batch_size=100, show_progress=False)

        mock_index.upsert.assert_called_once_with(
            vectors=data,
            namespace="test-ns",
            batch_size=100,
            show_progress=False,
        )

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_query_returns_none_when_no_index_selected(
        self, mock_weave_init, mock_pinecone_class
    ):
        """Test query returns None when no index is selected."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        db.index = None

        result = db.query(namespace="test-ns", vector=[0.1, 0.2])

        assert result is None

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_query_with_index_selected(self, mock_weave_init, mock_pinecone_class):
        """Test query works when index is selected."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        mock_index = mock.MagicMock()
        mock_index.query.return_value = {"matches": [{"id": "1", "score": 0.9}]}
        db.index = mock_index
        db.index_name = "test-index"

        result = db.query(
            namespace="test-ns",
            vector=[0.1, 0.2],
            top_k=10,
            sparse_vector={"indices": [1, 2], "values": [0.5, 0.5]},
            filter={"key": "value"},
            include_values=True,
            include_metadata=False,
        )

        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2],
            top_k=10,
            namespace="test-ns",
            sparse_vector={"indices": [1, 2], "values": [0.5, 0.5]},
            filter={"key": "value"},
            include_values=True,
            include_metadata=False,
        )
        assert result == {"matches": [{"id": "1", "score": 0.9}]}

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_delete_index_raises_valueerror_when_no_index_name(
        self, mock_weave_init, mock_pinecone_class
    ):
        """Test delete_index raises ValueError when no index_name."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        db.index_name = None

        with pytest.raises(ValueError, match="No index selected"):
            db.delete_index()

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_delete_index_deletes_index(self, mock_weave_init, mock_pinecone_class):
        """Test delete_index deletes the index properly."""
        mock_client = mock.MagicMock()
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key")
        db.index_name = "test-index"
        db.index = mock.MagicMock()

        db.delete_index()

        mock_client.delete_index.assert_called_once_with("test-index")
        assert db.index_name is None
        assert db.index is None

    @mock.patch("vectordb.pinecone.Pinecone")
    @mock.patch("vectordb.pinecone.weave.init")
    def test_init_selects_index_when_index_name_provided(
        self, mock_weave_init, mock_pinecone_class
    ):
        """Test that __init__ selects index when index_name is provided."""
        mock_client = mock.MagicMock()
        mock_client.list_indexes.return_value = [{"name": "my-index"}]
        mock_index = mock.MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        db = PineconeVectorDB(api_key="test-api-key", index_name="my-index")

        assert db.index_name == "my-index"
        assert db.index is mock_index
