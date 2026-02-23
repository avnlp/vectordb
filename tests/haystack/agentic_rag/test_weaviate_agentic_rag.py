"""Unit tests for WeaviateAgenticRAGPipeline class.

Tests all methods and functionality specific to the Weaviate implementation.
"""

from unittest.mock import Mock, patch

import pytest
from haystack import Document

from vectordb.haystack.agentic_rag.weaviate_agentic_rag import (
    WeaviateAgenticRAGPipeline,
)


@pytest.fixture
def mock_weaviate_config():
    """Mock configuration for Weaviate testing."""
    return {
        "agentic_rag": {
            "model": "test-model",
            "max_retries": 3,
            "retry_delay_seconds": 0.5,
            "fallback_tool": "retrieval",
            "routing_enabled": True,
            "self_reflection_enabled": False,
            "max_iterations": 2,
            "quality_threshold": 75,
        },
        "generator": {
            "model": "test-generator",
            "max_tokens": 2048,
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32,
        },
        "retrieval": {
            "top_k_default": 10,
        },
        "indexing": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "batch_size": 100,
        },
        "dataloader": {
            "type": "triviaqa",
            "dataset_name": "triviaqa",
            "split": "test",
            "limit": 10,
        },
        "weaviate": {
            "host": "localhost",
            "port": 8080,
            "grpc_port": 50051,
            "api_key": "test-api-key",
        },
        "collection": {"name": "TestCollection"},
    }


@pytest.fixture
def mock_weaviate_pipeline(mock_weaviate_config):
    """Create a mock Weaviate pipeline instance for testing."""
    # Create a mock pipeline instance without calling the actual constructor
    pipeline = WeaviateAgenticRAGPipeline.__new__(
        WeaviateAgenticRAGPipeline
    )  # Create without calling __init__

    # Set up the necessary attributes manually
    pipeline.config = mock_weaviate_config
    pipeline.logger = Mock()
    pipeline.dense_embedder = Mock()
    pipeline.document_embedder = Mock()
    pipeline.generator = Mock()
    pipeline.dataloader = Mock()
    pipeline.router = Mock()
    pipeline.data = None
    pipeline.documents = None
    pipeline.ground_truths = None
    pipeline.client = Mock()
    pipeline.collection_name = "TestCollection"

    return pipeline


class TestWeaviateAgenticRAGPipeline:
    """Unit tests for WeaviateAgenticRAGPipeline methods."""

    def test_connect_with_api_key(self, mock_weaviate_pipeline):
        """Test connecting to Weaviate with API key."""
        mock_client = Mock()
        mock_weaviate = Mock()
        mock_weaviate.auth.AuthApiKey.return_value = "auth"
        mock_weaviate.connect_to_custom.return_value = mock_client
        mock_lazy = Mock()
        mock_lazy_instance = Mock()
        mock_lazy_instance.__enter__ = Mock()
        mock_lazy_instance.__exit__ = Mock()
        mock_lazy.return_value = mock_lazy_instance
        mock_lazy_instance.__enter__.return_value = mock_lazy_instance
        mock_lazy_instance.__exit__.return_value = False
        mock_lazy_instance.check = Mock()

        with (
            patch.dict("sys.modules", {"weaviate": mock_weaviate}),
            patch(
                "haystack.lazy_imports.LazyImport",
                mock_lazy,
            ),
        ):
            mock_weaviate_pipeline._connect()

        mock_weaviate.auth.AuthApiKey.assert_called_once_with(api_key="test-api-key")
        mock_weaviate.connect_to_custom.assert_called_once_with(
            http_host="localhost",
            http_port=8080,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials="auth",
        )
        assert mock_weaviate_pipeline.client is mock_client
        mock_weaviate_pipeline.logger.info.assert_called_once_with(
            "Connected to Weaviate at %s:%s", "localhost", 8080
        )

    def test_connect_without_api_key(self, mock_weaviate_pipeline):
        """Test connecting to Weaviate without API key."""
        mock_client = Mock()
        mock_weaviate_pipeline.config["weaviate"].pop("api_key", None)
        mock_weaviate = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_lazy = Mock()
        mock_lazy_instance = Mock()
        mock_lazy_instance.__enter__ = Mock()
        mock_lazy_instance.__exit__ = Mock()
        mock_lazy.return_value = mock_lazy_instance
        mock_lazy_instance.__enter__.return_value = mock_lazy_instance
        mock_lazy_instance.__exit__.return_value = False
        mock_lazy_instance.check = Mock()

        with (
            patch.dict("sys.modules", {"weaviate": mock_weaviate}),
            patch(
                "haystack.lazy_imports.LazyImport",
                mock_lazy,
            ),
        ):
            mock_weaviate_pipeline._connect()

        mock_weaviate.connect_to_local.assert_called_once_with(
            host="localhost", port=8080, grpc_port=50051
        )
        assert mock_weaviate_pipeline.client is mock_client
        mock_weaviate_pipeline.logger.info.assert_called_once_with(
            "Connected to Weaviate at %s:%s", "localhost", 8080
        )

    def test_create_index_existing(self, mock_weaviate_pipeline):
        """Test creating Weaviate collection when it already exists."""
        mock_weaviate_pipeline.client.collections.exists.return_value = True

        mock_weaviate_pipeline._create_index()

        # Verify collection existence was checked
        mock_weaviate_pipeline.client.collections.exists.assert_called_once_with(
            "TestCollection"
        )
        assert mock_weaviate_pipeline.collection_name == "TestCollection"

    def test_create_index_nonexistent(self, mock_weaviate_pipeline):
        """Test creating Weaviate collection when it doesn't exist."""
        mock_weaviate_pipeline.client.collections.exists.return_value = False

        mock_weaviate_pipeline._create_index()

        # Should log warning but still set collection name
        assert mock_weaviate_pipeline.collection_name == "TestCollection"

    def test_create_index_with_exception(self, mock_weaviate_pipeline):
        """Test creating Weaviate collection when exception occurs."""
        mock_weaviate_pipeline.client.collections.exists.side_effect = Exception(
            "Check failed"
        )

        mock_weaviate_pipeline._create_index()

        # Should log warning but still set collection name
        assert mock_weaviate_pipeline.collection_name == "TestCollection"

    def test_index_documents_creates_collection_if_needed(self, mock_weaviate_pipeline):
        """Test indexing documents creates collection if it doesn't exist."""
        mock_doc1 = Document(
            content="Document 1", embedding=[0.1, 0.2, 0.3], meta={"source": "test1"}
        )
        mock_doc2 = Document(
            content="Document 2", embedding=[0.4, 0.5, 0.6], meta={"source": "test2"}
        )
        mock_weaviate_pipeline.embed_documents = Mock(
            return_value=[mock_doc1, mock_doc2]
        )

        # Mock collection existence check to return False
        mock_weaviate_pipeline.client.collections.exists.return_value = False

        # Mock the collection creation
        mock_weaviate_pipeline.client.collections.create = Mock()

        result = mock_weaviate_pipeline.index_documents()

        # Verify collection was created
        mock_weaviate_pipeline.client.collections.create.assert_called_once()
        args, kwargs = mock_weaviate_pipeline.client.collections.create.call_args
        assert kwargs["name"] == "TestCollection"
        assert result == 2

    def test_index_documents_with_no_documents(self, mock_weaviate_pipeline):
        """Test indexing documents when no documents are available."""
        mock_weaviate_pipeline.embed_documents = Mock(return_value=[])

        result = mock_weaviate_pipeline.index_documents()

        assert result == 0
        mock_weaviate_pipeline.logger.warning.assert_called_once_with(
            "No documents to index"
        )

    def test_index_documents_success(self, mock_weaviate_pipeline):
        """Test indexing documents successfully."""
        mock_doc1 = Document(
            content="Document 1", embedding=[0.1, 0.2, 0.3], meta={"source": "test1"}
        )
        mock_doc2 = Document(
            content="Document 2", embedding=[0.4, 0.5, 0.6], meta={"source": "test2"}
        )
        mock_weaviate_pipeline.embed_documents = Mock(
            return_value=[mock_doc1, mock_doc2]
        )

        # Mock collection existence check to return True
        mock_weaviate_pipeline.client.collections.exists.return_value = True

        # Mock collection object
        mock_collection = Mock()
        mock_weaviate_pipeline.client.collections.get.return_value = mock_collection

        result = mock_weaviate_pipeline.index_documents()

        # Verify documents were inserted into collection
        mock_collection.data.insert_many.assert_called_once()
        args, kwargs = mock_collection.data.insert_many.call_args
        objects = args[0]

        assert len(objects) == 2
        assert objects[0]["properties"]["content"] == "Document 1"
        assert objects[0]["properties"]["metadata"] == {"source": "test1"}
        assert objects[1]["properties"]["content"] == "Document 2"
        assert objects[1]["properties"]["metadata"] == {"source": "test2"}
        assert result == 2

    def test_index_documents_in_batches(self, mock_weaviate_pipeline):
        """Test indexing documents in batches."""
        # Create more documents than batch size to trigger batching
        documents = []
        for i in range(150):  # More than batch size of 100
            doc = Document(content=f"Document {i}", embedding=[0.1, 0.2, 0.3])
            documents.append(doc)
        mock_weaviate_pipeline.embed_documents = Mock(return_value=documents)

        # Mock collection existence check to return True
        mock_weaviate_pipeline.client.collections.exists.return_value = True

        # Mock collection object
        mock_collection = Mock()
        mock_weaviate_pipeline.client.collections.get.return_value = mock_collection

        result = mock_weaviate_pipeline.index_documents()

        # Verify multiple calls to insert_many for batching
        assert mock_collection.data.insert_many.call_count == 2  # Two batches
        assert result == 150

    def test_retrieve_success(self, mock_weaviate_pipeline):
        """Test retrieving documents successfully."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_weaviate_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock collection object
        mock_collection = Mock()
        mock_weaviate_pipeline.client.collections.get.return_value = mock_collection

        # Mock query results
        mock_obj1 = Mock()
        mock_obj1.properties = {"content": "Content 1", "field": "value1"}
        mock_obj1.metadata.distance = 0.1
        mock_obj2 = Mock()
        mock_obj2.properties = {"content": "Content 2", "field": "value2"}
        mock_obj2.metadata.distance = 0.2

        mock_response = Mock()
        mock_response.objects = [mock_obj1, mock_obj2]
        mock_collection.query.near_vector.return_value = mock_response

        documents = mock_weaviate_pipeline._retrieve(query, top_k)

        # Verify embedding was generated
        mock_weaviate_pipeline.dense_embedder.run.assert_called_once_with(text=query)

        # Verify query was called with correct parameters
        mock_collection.query.near_vector.assert_called_once_with(
            near_vector=mock_embedding,
            limit=top_k,
            return_metadata=["distance"],
        )

        # Verify returned documents
        assert len(documents) == 2
        assert documents[0].content == "Content 1"
        assert documents[0].meta == {"field": "value1"}
        assert documents[0].score == 0.9  # 1.0 - 0.1
        assert documents[1].content == "Content 2"
        assert documents[1].meta == {"field": "value2"}
        assert documents[1].score == 0.8  # 1.0 - 0.2

    def test_retrieve_with_missing_distance_metadata(self, mock_weaviate_pipeline):
        """Test retrieving documents when distance metadata is missing."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_weaviate_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock collection object
        mock_collection = Mock()
        mock_weaviate_pipeline.client.collections.get.return_value = mock_collection

        # Mock query results with missing distance metadata
        mock_obj = Mock()
        mock_obj.properties = {"content": "Content 1", "field": "value1"}
        mock_obj.metadata = None  # No metadata

        mock_response = Mock()
        mock_response.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_response

        documents = mock_weaviate_pipeline._retrieve(query, top_k)

        # Verify document has score of 1.0 when metadata is None
        assert len(documents) == 1
        assert documents[0].score == 1.0  # 1.0 - 0.0 (default)

    def test_retrieve_with_empty_properties(self, mock_weaviate_pipeline):
        """Test retrieving documents when properties are empty."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_weaviate_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock collection object
        mock_collection = Mock()
        mock_weaviate_pipeline.client.collections.get.return_value = mock_collection

        # Mock query results with empty properties
        mock_obj = Mock()
        mock_obj.properties = {}  # Empty properties
        mock_obj.metadata.distance = 0.1

        mock_response = Mock()
        mock_response.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_response

        documents = mock_weaviate_pipeline._retrieve(query, top_k)

        # Verify document has empty content when properties are empty
        assert len(documents) == 1
        assert documents[0].content == ""
        assert documents[0].meta == {}

    def test_retrieve_with_exception(self, mock_weaviate_pipeline):
        """Test retrieving documents when an exception occurs."""
        query = "test query"
        top_k = 5

        mock_collection = Mock()
        mock_weaviate_pipeline.client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.side_effect = Exception("Query failed")

        documents = mock_weaviate_pipeline._retrieve(query, top_k)

        # Should return empty list on exception
        assert documents == []
        mock_weaviate_pipeline.logger.error.assert_called_once()
