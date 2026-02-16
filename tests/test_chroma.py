"""Tests for the Chroma vector database interface."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock external modules at the very beginning before any imports
sys.modules["pysqlite3"] = MagicMock()
sys.modules["sqlite3"] = MagicMock()


# Create a proper mock for weave that includes Model base class
class MockWeaveModel:
    """Mock base class for weave.Model."""

    def __init__(self, **kwargs):
        # Copy class attributes to instance
        for attr_name in dir(self.__class__):
            if not attr_name.startswith("_"):
                attr_value = getattr(self.__class__, attr_name)
                if not callable(attr_value):
                    setattr(self, attr_name, attr_value)
        # Set instance attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockWeave:
    """Mock weave module."""

    class Model(MockWeaveModel):
        """Mock Model class."""

    @staticmethod
    def op():
        """Mock op function."""

        def decorator(func):
            return func

        return decorator

    @staticmethod
    def model(cls_):
        """Decorator to mark a class as a weave model."""
        return cls_

    @staticmethod
    def init(project_name, **kwargs):
        """Mock init function."""
        pass

    @staticmethod
    def track(value, name=None):
        """Mock track function for weave tracking."""
        return value


sys.modules["weave"] = MockWeave()

# Create mock for chromadb and its submodules
mock_chromadb = MagicMock()
sys.modules["chromadb"] = mock_chromadb
sys.modules["chromadb.api"] = MagicMock()
sys.modules["chromadb.api.configuration"] = MagicMock()
sys.modules["chromadb.api.types"] = MagicMock()
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()

# Import after setting up mocks
from vectordb.chroma import ChromaVectorDB  # noqa: E402


class TestChromaVectorDB:
    """Test cases for ChromaVectorDB class."""

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_init_with_default_parameters(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test initialization with default parameters."""
        mock_client = MagicMock()
        mock_persistent_client_class.return_value = mock_client
        mock_client_class.return_value = mock_client

        db = ChromaVectorDB()

        assert db.path == "./chroma"
        assert db.persistent is True
        assert db.collection_name is None
        assert db.tracing_project_name == "chroma"
        assert db.weave_params is None
        assert db.client == mock_client
        assert db.collection is None
        mock_persistent_client_class.assert_called_once_with("./chroma")

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_init_with_non_persistent_client(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test initialization with non-persistent client."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        db = ChromaVectorDB(persistent=False)

        assert db.persistent is False
        mock_client_class.assert_called_once()
        mock_persistent_client_class.assert_not_called()

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_init_with_existing_collection_name(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test initialization with existing collection_name calls get_collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_persistent_client_class.return_value = mock_client

        db = ChromaVectorDB(collection_name="test_collection")

        assert db.collection_name == "test_collection"
        assert db.collection == mock_collection
        mock_client.get_collection.assert_called_once_with("test_collection")

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_init_with_custom_parameters(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test initialization with custom parameters."""
        mock_client = MagicMock()
        mock_persistent_client_class.return_value = mock_client
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

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_create_collection_calls_get_or_create_collection(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test that create_collection calls get_or_create_collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client_class.return_value = mock_client

        # Mock the embedding functions module
        import chromadb.utils.embedding_functions as ef_module

        ef_module.DefaultEmbeddingFunction = MagicMock

        db = ChromaVectorDB()
        db.create_collection(name="test_collection")

        assert db.collection == mock_collection
        mock_client.get_or_create_collection.assert_called_once()
        call_args = mock_client.get_or_create_collection.call_args
        assert call_args.kwargs["name"] == "test_collection"

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_create_collection_with_custom_parameters(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test create_collection with custom parameters."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client_class.return_value = mock_client

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

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_upsert_raises_error_when_collection_is_none(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test that upsert raises AttributeError when collection is None."""
        mock_client = MagicMock()
        mock_persistent_client_class.return_value = mock_client

        db = ChromaVectorDB()
        # collection is None by default

        data = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "texts": ["test text"],
            "metadatas": [{"key": "value"}],
            "ids": ["id1"],
        }

        # When collection is None, calling .add() raises AttributeError
        with pytest.raises(AttributeError):
            db.upsert(data)

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_upsert_success(self, mock_client_class, mock_persistent_client_class):
        """Test successful upsert operation."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_persistent_client_class.return_value = mock_client

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

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_upsert_with_additional_kwargs(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test upsert with additional keyword arguments."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_persistent_client_class.return_value = mock_client

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

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_query_raises_error_when_collection_is_none(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test that query raises AttributeError when collection is None."""
        mock_client = MagicMock()
        mock_persistent_client_class.return_value = mock_client

        db = ChromaVectorDB()
        # collection is None by default

        # When collection is None, calling .query() raises AttributeError
        with pytest.raises(AttributeError):
            db.query(query_embedding=[0.1, 0.2, 0.3])

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_query_success(self, mock_client_class, mock_persistent_client_class):
        """Test successful query operation."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_query_result = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
        }
        mock_collection.query.return_value = mock_query_result
        mock_persistent_client_class.return_value = mock_client

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

    @patch("chromadb.PersistentClient")
    @patch("chromadb.Client")
    def test_query_with_custom_parameters(
        self, mock_client_class, mock_persistent_client_class
    ):
        """Test query with custom parameters."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_query_result = {"ids": [["id1"]]}
        mock_collection.query.return_value = mock_query_result
        mock_persistent_client_class.return_value = mock_client

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
