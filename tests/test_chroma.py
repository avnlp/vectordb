"""Tests for the Chroma vector database interface."""

import sys
from unittest.mock import MagicMock

import pytest

# NOTE: Mocks are now set up in conftest.py via pytest_configure
# to ensure they work properly with pytest-xdist
# Import after setting up mocks
from vectordb.chroma import ChromaVectorDB  # noqa: E402


@pytest.fixture
def chromadb_mocked():
    """Fixture to set up chromadb mocking for each test."""
    # Create fresh mocks for this test
    mock_client = MagicMock()
    mock_persistent_client = MagicMock()

    # Set up the mocks in sys.modules
    sys.modules["chromadb"].Client = MagicMock(return_value=mock_client)
    sys.modules["chromadb"].PersistentClient = MagicMock(
        return_value=mock_persistent_client
    )
    sys.modules[
        "chromadb.utils.embedding_functions"
    ].DefaultEmbeddingFunction = MagicMock

    yield {
        "client": mock_client,
        "persistent_client": mock_persistent_client,
    }


@pytest.mark.usefixtures("chromadb_mocked")
class TestChromaVectorDB:
    """Test cases for ChromaVectorDB class."""

    def test_init_with_default_parameters(self, chromadb_mocked):
        """Test initialization with default parameters."""
        mock_persistent_client_class = sys.modules["chromadb"].PersistentClient

        db = ChromaVectorDB()

        assert db.path == "./chroma"
        assert db.persistent is True
        assert db.collection_name is None
        assert db.tracing_project_name == "chroma"
        assert db.weave_params is None
        assert db.collection is None
        mock_persistent_client_class.assert_called_once_with("./chroma")

    def test_init_with_non_persistent_client(self, chromadb_mocked):
        """Test initialization with non-persistent client."""
        mock_client_class = sys.modules["chromadb"].Client
        mock_persistent_client_class = sys.modules["chromadb"].PersistentClient

        db = ChromaVectorDB(persistent=False)

        assert db.persistent is False
        mock_client_class.assert_called_once()
        mock_persistent_client_class.assert_not_called()

    def test_init_with_existing_collection_name(self, chromadb_mocked):
        """Test initialization with existing collection_name calls get_collection."""
        # Reset and set up fresh mocks for this test
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        sys.modules["chromadb"].Client = MagicMock(return_value=mock_client)
        sys.modules["chromadb"].PersistentClient = MagicMock(return_value=mock_client)

        db = ChromaVectorDB(collection_name="test_collection")

        assert db.collection_name == "test_collection"
        assert db.collection == mock_collection
        mock_client.get_collection.assert_called_once_with("test_collection")

    def test_init_with_custom_parameters(self, chromadb_mocked):
        """Test initialization with custom parameters."""
        mock_persistent_client_class = sys.modules["chromadb"].PersistentClient
        weave_params = {"param1": "value1"}

        db = ChromaVectorDB(
            path="/custom/path",
            persistent=True,
            tracing_project_name="custom_project",
            weave_params=weave_params,
        )

        assert db.path == "/custom/path"
        assert db.tracing_project_name == "custom_project"
        assert db.weave_params == weave_params
        mock_persistent_client_class.assert_called_once_with("/custom/path")

    def test_create_collection_calls_get_or_create_collection(self, chromadb_mocked):
        """Test that create_collection calls get_or_create_collection."""
        # Reset and set up fresh mocks for this test
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        sys.modules["chromadb"].Client = MagicMock(return_value=mock_client)
        sys.modules["chromadb"].PersistentClient = MagicMock(return_value=mock_client)

        db = ChromaVectorDB()
        db.create_collection(name="test_collection")

        assert db.collection == mock_collection
        mock_client.get_or_create_collection.assert_called_once()
        call_args = mock_client.get_or_create_collection.call_args
        assert call_args.kwargs["name"] == "test_collection"

    def test_create_collection_with_custom_parameters(self, chromadb_mocked):
        """Test create_collection with custom parameters."""
        # Reset and set up fresh mocks for this test
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        sys.modules["chromadb"].Client = MagicMock(return_value=mock_client)
        sys.modules["chromadb"].PersistentClient = MagicMock(return_value=mock_client)

        mock_embedding_function = MagicMock()
        mock_configuration = MagicMock()
        mock_metadata = {"key": "value"}

        db = ChromaVectorDB()
        db.create_collection(
            name="test_collection",
            configuration=mock_configuration,
            metadata=mock_metadata,
            embedding_function=mock_embedding_function,
            custom_arg="custom_value",
        )

        mock_client.get_or_create_collection.assert_called_once_with(
            name="test_collection",
            configuration=mock_configuration,
            metadata=mock_metadata,
            embedding_function=mock_embedding_function,
            custom_arg="custom_value",
        )

    def test_upsert_raises_error_when_collection_is_none(self, chromadb_mocked):
        """Test that upsert raises ValueError when collection is None."""
        db = ChromaVectorDB()
        # collection is None by default

        data = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "texts": ["test text"],
            "metadatas": [{"key": "value"}],
            "ids": ["id1"],
        }

        # When collection is None, calling upsert raises ValueError
        with pytest.raises(ValueError, match="No collection initialized"):
            db.upsert(data)

    def test_upsert_success(self, chromadb_mocked):
        """Test successful upsert operation."""
        mock_collection = MagicMock()

        db = ChromaVectorDB()
        db.collection = mock_collection

        data = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "texts": ["text1", "text2"],
            "metadatas": [{"key": "val1"}, {"key": "val2"}],
            "ids": ["id1", "id2"],
        }

        db.upsert(data)

        mock_collection.add.assert_called_once_with(
            embeddings=data["embeddings"],
            documents=data["texts"],
            metadatas=data["metadatas"],
            ids=data["ids"],
        )

    def test_upsert_with_additional_kwargs(self, chromadb_mocked):
        """Test upsert with additional keyword arguments."""
        mock_collection = MagicMock()

        db = ChromaVectorDB()
        db.collection = mock_collection

        data = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "texts": ["text1"],
            "metadatas": [{"key": "val1"}],
            "ids": ["id1"],
        }

        db.upsert(data, custom_param="custom_value")

        mock_collection.add.assert_called_once_with(
            embeddings=data["embeddings"],
            documents=data["texts"],
            metadatas=data["metadatas"],
            ids=data["ids"],
            custom_param="custom_value",
        )

    def test_query_raises_error_when_collection_is_none(self, chromadb_mocked):
        """Test that query raises ValueError when collection is None."""
        db = ChromaVectorDB()
        # collection is None by default

        # When collection is None, calling query raises ValueError
        with pytest.raises(ValueError, match="No collection initialized"):
            db.query(query_embedding=[0.1, 0.2, 0.3])

    def test_query_success(self, chromadb_mocked):
        """Test successful query operation."""
        mock_collection = MagicMock()
        mock_query_result = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
        }
        mock_collection.query.return_value = mock_query_result

        db = ChromaVectorDB()
        db.collection = mock_collection

        result = db.query(query_embedding=[0.1, 0.2, 0.3])

        assert result == mock_query_result
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=10,
            where=None,
            where_document=None,
        )

    def test_query_with_custom_parameters(self, chromadb_mocked):
        """Test query with custom parameters."""
        mock_collection = MagicMock()
        mock_query_result = {"ids": [["id1"]]}
        mock_collection.query.return_value = mock_query_result

        db = ChromaVectorDB()
        db.collection = mock_collection

        where_filter = {"key": "value"}
        where_document_filter = {"$contains": "search term"}

        result = db.query(
            query_embedding=[0.1, 0.2, 0.3],
            n_results=5,
            where=where_filter,
            where_document=where_document_filter,
            custom_param="value",
        )

        assert result == mock_query_result
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=5,
            where=where_filter,
            where_document=where_document_filter,
            custom_param="value",
        )
