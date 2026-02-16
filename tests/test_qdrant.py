"""Tests for the Qdrant vector database interface."""

from unittest import mock

import pytest


class MockVectorParams:
    """Mock VectorParams class."""

    def __init__(self, size, distance):
        self.size = size
        self.distance = distance


class MockPointStruct:
    """Mock PointStruct class."""

    def __init__(self, id, vector, payload=None):
        self.id = id
        self.vector = vector
        self.payload = payload


class MockScoredPoint:
    """Mock ScoredPoint class."""

    def __init__(self, id, score, payload=None):
        self.id = id
        self.score = score
        self.payload = payload


class MockFilter:
    """Mock Filter class."""

    pass


class MockQdrantClient:
    """Mock QdrantClient class."""

    def __init__(self, host, port, api_key=None, timeout=60.0):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.timeout = timeout

    def get_collection(self, collection_name, raise_on_not_found=False):
        """Mock get_collection method."""
        return

    def create_collection(self, collection_name, vectors_config):
        """Mock create_collection method."""
        pass

    def upsert(self, collection_name, points):
        """Mock upsert method."""
        pass

    def search(self, collection_name, query_vector, limit, query_filter, with_payload):
        """Mock search method."""
        return []

    def delete_collection(self, collection_name):
        """Mock delete_collection method."""
        pass

    def get_collections(self):
        """Mock get_collections method."""
        mock_collection = mock.MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = mock.MagicMock()
        mock_collections.collections = [mock_collection]
        return mock_collections


@pytest.fixture
def mock_qdrant_client():
    """Fixture to mock qdrant_client module."""
    with mock.patch.dict(
        "sys.modules",
        {
            "qdrant_client": mock.MagicMock(
                QdrantClient=MockQdrantClient,
            ),
            "qdrant_client.http.models": mock.MagicMock(
                VectorParams=MockVectorParams,
                PointStruct=MockPointStruct,
                Filter=MockFilter,
                ScoredPoint=MockScoredPoint,
            ),
        },
    ):
        from vectordb.qdrant import QdrantVectorDB

        yield QdrantVectorDB


@pytest.fixture
def mock_qdrant_instance(mock_qdrant_client):
    """Fixture to create a QdrantVectorDB instance with mocked client."""
    return mock_qdrant_client(
        host="localhost", port=6333, api_key=None, collection_name="test_collection"
    )


class TestQdrantVectorDB:
    """Test cases for QdrantVectorDB class."""

    def test_initialization_creates_qdrant_client(self, mock_qdrant_client):
        """Test that initialization creates QdrantClient with correct parameters."""
        db = mock_qdrant_client(
            host="test_host", port=8080, api_key="test_key", timeout=30.0
        )

        assert db.client.host == "test_host"
        assert db.client.port == 8080
        assert db.client.api_key == "test_key"
        assert db.client.timeout == 30.0

    def test_create_collection_skips_if_already_exists(self, mock_qdrant_instance):
        """Test that create_collection skips if collection already exists."""
        mock_qdrant_instance.client.get_collection = mock.MagicMock(return_value=True)
        mock_qdrant_instance.client.create_collection = mock.MagicMock()

        mock_qdrant_instance.create_collection(
            collection_name="existing_collection", vector_size=128, distance="Cosine"
        )

        mock_qdrant_instance.client.get_collection.assert_called_once_with(
            "existing_collection", raise_on_not_found=False
        )
        mock_qdrant_instance.client.create_collection.assert_not_called()
        assert mock_qdrant_instance.collection_name == "existing_collection"

    def test_create_collection_creates_with_vector_params(self, mock_qdrant_instance):
        """Test that create_collection creates collection with VectorParams."""
        mock_qdrant_instance.client.get_collection = mock.MagicMock(return_value=None)
        mock_qdrant_instance.client.create_collection = mock.MagicMock()

        mock_qdrant_instance.create_collection(
            collection_name="new_collection", vector_size=256, distance="Euclidean"
        )

        mock_qdrant_instance.client.create_collection.assert_called_once()
        call_args = mock_qdrant_instance.client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "new_collection"
        assert call_args.kwargs["vectors_config"].size == 256
        assert call_args.kwargs["vectors_config"].distance == "Euclidean"
        assert mock_qdrant_instance.collection_name == "new_collection"

    def test_upsert_vectors_raises_value_error_when_no_collection_name(
        self, mock_qdrant_client
    ):
        """Test that upsert_vectors raises ValueError when no collection_name."""
        db = mock_qdrant_client()

        with pytest.raises(ValueError, match="No collection selected"):
            db.upsert_vectors(vectors=[])

    def test_query_vectors_raises_value_error_when_no_collection_name(
        self, mock_qdrant_client
    ):
        """Test that query_vectors raises ValueError when no collection_name."""
        db = mock_qdrant_client()

        with pytest.raises(ValueError, match="No collection selected"):
            db.query_vectors(query_vector=[0.1, 0.2, 0.3])

    def test_delete_collection_raises_value_error_when_no_collection_name(
        self, mock_qdrant_client
    ):
        """Test that delete_collection raises ValueError when no collection_name."""
        db = mock_qdrant_client()

        with pytest.raises(ValueError, match="No collection selected"):
            db.delete_collection()

    def test_list_collections_returns_collection_names(self, mock_qdrant_instance):
        """Test that list_collections returns collection names."""
        collections = mock_qdrant_instance.list_collections()

        assert collections == ["test_collection"]
