"""Tests for the Milvus VectorDB module."""

import sys
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest


# Pre-load mocks in sys.modules BEFORE any test imports the milvus module
# This ensures the mocks are available when the module is first imported

# Create weave mock with all required attributes
_weave_mock = mock.MagicMock()
_weave_mock.model = lambda cls: cls  # type: ignore[misc]
_weave_mock.op = lambda: lambda func: func  # type: ignore[misc]
_weave_mock.track = lambda value, name=None: value  # type: ignore[misc]
_weave_mock.init = mock.MagicMock()
sys.modules["weave"] = _weave_mock

# Create pymilvus mock with MagicMock that acts as callable classes
_pymilvus_mock = mock.MagicMock()
sys.modules["pymilvus"] = _pymilvus_mock

# Now we can safely import MilvusVectorDB - the imports will use our mocks
from vectordb.milvus import MilvusVectorDB  # noqa: E402


class TestMilvusVectorDB:
    """Test cases for MilvusVectorDB class."""

    def test_init_connects_to_milvus(self):
        """Test that initialization connects to Milvus server."""
        _pymilvus_mock.connections.connect = mock.MagicMock()

        db = MilvusVectorDB(
            host="localhost", port="19530", collection_name="test-collection"
        )

        _pymilvus_mock.connections.connect.assert_called_once_with(
            alias="default", host="localhost", port="19530"
        )
        assert db.host == "localhost"
        assert db.port == "19530"
        assert db.collection_name == "test-collection"
        assert db.collection is None

    def test_create_collection_creates_schema_with_proper_fields(self):
        """Test create_collection creates schema with proper fields."""
        mock_int64_type = MagicMock()
        mock_float_vector_type = MagicMock()
        _pymilvus_mock.DataType.INT64 = mock_int64_type
        _pymilvus_mock.DataType.FLOAT_VECTOR = mock_float_vector_type

        mock_id_field = MagicMock()
        mock_vector_field = MagicMock()
        _pymilvus_mock.FieldSchema.side_effect = [mock_id_field, mock_vector_field]

        mock_schema = MagicMock()
        _pymilvus_mock.CollectionSchema.return_value = mock_schema

        mock_collection_instance = MagicMock()
        _pymilvus_mock.Collection.return_value = mock_collection_instance

        db = MilvusVectorDB(host="localhost", port="19530")
        db.create_collection(
            collection_name="test-collection",
            dimension=128,
            metric_type="L2",
            description="Test collection",
        )

        assert _pymilvus_mock.FieldSchema.call_count == 2
        _pymilvus_mock.FieldSchema.assert_any_call(
            name="id", dtype=mock_int64_type, is_primary=True, auto_id=True
        )
        _pymilvus_mock.FieldSchema.assert_any_call(
            name="vector", dtype=mock_float_vector_type, dim=128
        )

        _pymilvus_mock.CollectionSchema.assert_called_once_with(
            [mock_id_field, mock_vector_field], description="Test collection"
        )
        _pymilvus_mock.Collection.assert_called_once_with(
            name="test-collection", schema=mock_schema
        )
        assert db.collection is mock_collection_instance

    def test_insert_vectors_raises_valueerror_when_no_collection(self):
        """Test insert_vectors raises ValueError when no collection is selected."""
        db = MilvusVectorDB(host="localhost", port="19530")
        db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            db.insert_vectors([[0.1, 0.2, 0.3]], ids=[1])

    def test_query_vectors_raises_valueerror_when_no_collection(self):
        """Test query_vectors raises ValueError when no collection is selected."""
        db = MilvusVectorDB(host="localhost", port="19530")
        db.collection = None

        with pytest.raises(ValueError, match="No collection selected"):
            db.query_vectors([0.1, 0.2, 0.3], top_k=5)

    def test_delete_collection_raises_valueerror_when_no_collection_name(self):
        """Test delete_collection raises ValueError when no collection_name."""
        db = MilvusVectorDB(host="localhost", port="19530")
        db.collection_name = None

        with pytest.raises(ValueError, match="No collection specified"):
            db.delete_collection()

    def test_list_collections_returns_list_of_collection_names(self):
        """Test list_collections returns list of collection names."""
        _pymilvus_mock.connections.list_collections.return_value = [
            "collection1",
            "collection2",
            "collection3",
        ]

        db = MilvusVectorDB(host="localhost", port="19530")
        result = db.list_collections()

        _pymilvus_mock.connections.list_collections.assert_called_once()
        assert result == ["collection1", "collection2", "collection3"]

    def test_insert_vectors_with_collection_selected(self):
        """Test insert_vectors works when collection is selected."""
        db = MilvusVectorDB(host="localhost", port="19530")
        mock_collection = MagicMock()
        db.collection = mock_collection
        db.collection_name = "test-collection"

        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        ids = [1, 2]
        db.insert_vectors(vectors, ids=ids)

        expected_entities = {"id": [1, 2], "vector": vectors}
        mock_collection.insert.assert_called_once_with([expected_entities])

    def test_insert_vectors_without_ids_generates_default_ids(self):
        """Test insert_vectors generates default IDs when not provided."""
        db = MilvusVectorDB(host="localhost", port="19530")
        mock_collection = MagicMock()
        db.collection = mock_collection
        db.collection_name = "test-collection"

        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        db.insert_vectors(vectors)

        expected_entities = {"id": [0, 1, 2], "vector": vectors}
        mock_collection.insert.assert_called_once_with([expected_entities])

    def test_query_vectors_with_collection_selected(self):
        """Test query_vectors works when collection is selected."""
        db = MilvusVectorDB(host="localhost", port="19530")
        mock_collection = MagicMock()
        mock_results = MagicMock()
        mock_collection.search.return_value = mock_results
        db.collection = mock_collection
        db.collection_name = "test-collection"

        query_vector = [0.1, 0.2, 0.3]
        result = db.query_vectors(query_vector, top_k=10, metric_type="L2")

        expected_search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        mock_collection.search.assert_called_once_with(
            data=[query_vector],
            anns_field="vector",
            param=expected_search_params,
            limit=10,
        )
        assert result is mock_results

    def test_delete_collection_deletes_collection(self):
        """Test delete_collection properly deletes the collection."""
        db = MilvusVectorDB(host="localhost", port="19530")
        mock_collection = MagicMock()
        db.collection = mock_collection
        db.collection_name = "test-collection"

        db.delete_collection()

        mock_collection.drop.assert_called_once()
        assert db.collection is None
        assert db.collection_name is None

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        _pymilvus_mock.connections.connect = mock.MagicMock()

        db = MilvusVectorDB()

        _pymilvus_mock.connections.connect.assert_called_once_with(
            alias="default", host="localhost", port="19530"
        )
        assert db.host == "localhost"
        assert db.port == "19530"
        assert db.collection_name is None

    def test_create_collection_with_default_metric_type(self):
        """Test create_collection uses default metric_type when not specified."""
        import vectordb.milvus as milvus_module

        mock_collection_instance = MagicMock()
        mock_field_schema = MagicMock()
        mock_collection_schema = MagicMock()

        # Patch the module-level names directly
        with (
            patch.object(
                milvus_module, "Collection", return_value=mock_collection_instance
            ) as mock_col,
            patch.object(milvus_module, "FieldSchema", return_value=mock_field_schema),
            patch.object(
                milvus_module, "CollectionSchema", return_value=mock_collection_schema
            ),
        ):
            db = MilvusVectorDB(host="localhost", port="19530")
            db.create_collection(collection_name="test-collection", dimension=128)

            mock_col.assert_called_once()
