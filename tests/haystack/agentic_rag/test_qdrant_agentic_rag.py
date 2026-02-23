"""Unit tests for QdrantAgenticRAGPipeline class.

Tests all methods and functionality specific to the Qdrant implementation.
"""

from unittest.mock import Mock, patch

import pytest
from haystack import Document

from vectordb.haystack.agentic_rag.qdrant_agentic_rag import QdrantAgenticRAGPipeline


@pytest.fixture
def mock_qdrant_config():
    """Mock configuration for Qdrant testing."""
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
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "api_key": "test-api-key",
        },
        "collection": {"name": "test-collection"},
    }


@pytest.fixture
def mock_qdrant_pipeline(mock_qdrant_config):
    """Create a mock Qdrant pipeline instance for testing."""
    # Create a mock pipeline instance without calling the actual constructor
    pipeline = QdrantAgenticRAGPipeline.__new__(
        QdrantAgenticRAGPipeline
    )  # Create without calling __init__

    # Set up the necessary attributes manually
    pipeline.config = mock_qdrant_config
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
    pipeline.collection_name = "test-collection"

    return pipeline


class TestQdrantAgenticRAGPipeline:
    """Unit tests for QdrantAgenticRAGPipeline methods."""

    def test_connect_success(self, mock_qdrant_pipeline):
        """Test connecting to Qdrant successfully."""
        mock_client = Mock()

        with patch(
            "vectordb.haystack.agentic_rag.qdrant_agentic_rag.QdrantClient",
            return_value=mock_client,
        ) as mock_qdrant_client:
            mock_qdrant_pipeline._connect()

        mock_qdrant_client.assert_called_once_with(
            url="http://localhost:6333", api_key="test-api-key"
        )
        assert mock_qdrant_pipeline.client is mock_client
        mock_qdrant_pipeline.logger.info.assert_called_once_with(
            "Connected to Qdrant at %s:%s", "localhost", 6333
        )

    def test_connect_without_api_key(self, mock_qdrant_pipeline):
        """Test connecting to Qdrant without API key."""
        mock_client = Mock()
        mock_qdrant_pipeline.config["qdrant"]["api_key"] = None

        with patch(
            "vectordb.haystack.agentic_rag.qdrant_agentic_rag.QdrantClient",
            return_value=mock_client,
        ) as mock_qdrant_client:
            mock_qdrant_pipeline._connect()

        mock_qdrant_client.assert_called_once_with(
            url="http://localhost:6333", api_key=None
        )
        assert mock_qdrant_pipeline.client is mock_client
        mock_qdrant_pipeline.logger.info.assert_called_once_with(
            "Connected to Qdrant at %s:%s", "localhost", 6333
        )

    def test_create_index_existing(self, mock_qdrant_pipeline):
        """Test creating Qdrant collection when it already exists."""
        mock_collection_info = Mock()
        mock_collection_info.points_count = 500
        mock_qdrant_pipeline.client.get_collection.return_value = mock_collection_info

        mock_qdrant_pipeline._create_index()

        # Verify collection was checked
        mock_qdrant_pipeline.client.get_collection.assert_called_once_with(
            "test-collection"
        )
        assert mock_qdrant_pipeline.collection_name == "test-collection"

    def test_create_index_nonexistent(self, mock_qdrant_pipeline):
        """Test creating Qdrant collection when it doesn't exist."""
        mock_qdrant_pipeline.client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        mock_qdrant_pipeline._create_index()

        # Should log warning but still set collection name
        assert mock_qdrant_pipeline.collection_name == "test-collection"

    def test_index_documents_creates_collection_if_needed(self, mock_qdrant_pipeline):
        """Test indexing documents creates collection if it doesn't exist."""
        mock_doc1 = Document(content="Document 1", embedding=[0.1, 0.2, 0.3])
        mock_doc2 = Document(content="Document 2", embedding=[0.4, 0.5, 0.6])
        mock_qdrant_pipeline.embed_documents = Mock(return_value=[mock_doc1, mock_doc2])

        # Mock collection existence check to fail
        mock_qdrant_pipeline.client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        # Mock the collection creation
        mock_qdrant_pipeline.client.create_collection = Mock()

        result = mock_qdrant_pipeline.index_documents()

        # Verify collection was created
        mock_qdrant_pipeline.client.create_collection.assert_called_once()
        args, kwargs = mock_qdrant_pipeline.client.create_collection.call_args
        assert kwargs["collection_name"] == "test-collection"
        assert result == 2

    def test_index_documents_with_no_documents(self, mock_qdrant_pipeline):
        """Test indexing documents when no documents are available."""
        mock_qdrant_pipeline.embed_documents = Mock(return_value=[])

        result = mock_qdrant_pipeline.index_documents()

        assert result == 0
        mock_qdrant_pipeline.logger.warning.assert_called_once_with(
            "No documents to index"
        )

    def test_index_documents_success(self, mock_qdrant_pipeline):
        """Test indexing documents successfully."""
        mock_doc1 = Document(
            content="Document 1", embedding=[0.1, 0.2, 0.3], meta={"source": "test1"}
        )
        mock_doc2 = Document(
            content="Document 2", embedding=[0.4, 0.5, 0.6], meta={"source": "test2"}
        )
        mock_qdrant_pipeline.embed_documents = Mock(return_value=[mock_doc1, mock_doc2])

        # Mock collection existence check to succeed
        mock_collection_info = Mock()
        mock_qdrant_pipeline.client.get_collection.return_value = mock_collection_info

        result = mock_qdrant_pipeline.index_documents()

        # Verify documents were upserted to collection
        mock_qdrant_pipeline.client.upsert.assert_called_once()
        args, kwargs = mock_qdrant_pipeline.client.upsert.call_args
        assert kwargs["collection_name"] == "test-collection"

        points = kwargs["points"]
        assert len(points) == 2
        assert points[0].id == 0
        assert points[0].vector == [0.1, 0.2, 0.3]
        assert points[0].payload == {
            "content": "Document 1",
            "metadata": {"source": "test1"},
        }
        assert points[1].id == 1
        assert points[1].vector == [0.4, 0.5, 0.6]
        assert points[1].payload == {
            "content": "Document 2",
            "metadata": {"source": "test2"},
        }
        assert result == 2

    def test_index_documents_in_batches(self, mock_qdrant_pipeline):
        """Test indexing documents in batches."""
        # Create more documents than batch size to trigger batching
        documents = []
        for i in range(150):  # More than batch size of 100
            doc = Document(content=f"Document {i}", embedding=[0.1, 0.2, 0.3])
            documents.append(doc)
        mock_qdrant_pipeline.embed_documents = Mock(return_value=documents)

        # Mock collection existence check to succeed
        mock_collection_info = Mock()
        mock_qdrant_pipeline.client.get_collection.return_value = mock_collection_info

        result = mock_qdrant_pipeline.index_documents()

        # Verify multiple calls to upsert for batching
        assert mock_qdrant_pipeline.client.upsert.call_count == 2  # Two batches
        assert result == 150

    def test_retrieve_success(self, mock_qdrant_pipeline):
        """Test retrieving documents successfully."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_qdrant_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock search results - note that in the actual implementation,
        # metadata comes from result.payload.get("metadata", {})
        mock_result1 = Mock()
        mock_result1.payload = {"content": "Content 1", "metadata": {"field": "value1"}}
        mock_result1.score = 0.9
        mock_result2 = Mock()
        mock_result2.payload = {"content": "Content 2", "metadata": {"field": "value2"}}
        mock_result2.score = 0.8

        mock_qdrant_pipeline.client.search.return_value = [mock_result1, mock_result2]

        documents = mock_qdrant_pipeline._retrieve(query, top_k)

        # Verify embedding was generated
        mock_qdrant_pipeline.dense_embedder.run.assert_called_once_with(text=query)

        # Verify search was called with correct parameters
        mock_qdrant_pipeline.client.search.assert_called_once_with(
            collection_name="test-collection",
            query_vector=mock_embedding,
            limit=top_k,
        )

        # Verify returned documents
        assert len(documents) == 2
        assert documents[0].content == "Content 1"
        assert documents[0].meta == {"field": "value1"}
        assert documents[0].score == 0.9
        assert documents[1].content == "Content 2"
        assert documents[1].meta == {"field": "value2"}
        assert documents[1].score == 0.8

    def test_retrieve_with_empty_results(self, mock_qdrant_pipeline):
        """Test retrieving documents with empty results."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_qdrant_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock empty search results
        mock_qdrant_pipeline.client.search.return_value = []

        documents = mock_qdrant_pipeline._retrieve(query, top_k)

        assert len(documents) == 0

    def test_retrieve_with_missing_payload_fields(self, mock_qdrant_pipeline):
        """Test retrieving documents when payload fields are missing."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_qdrant_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock search result with missing content and metadata in payload
        mock_result = Mock()
        mock_result.payload = {"field": "value"}  # No 'content' or 'metadata' key
        mock_result.score = 0.9

        mock_qdrant_pipeline.client.search.return_value = [mock_result]

        documents = mock_qdrant_pipeline._retrieve(query, top_k)

        # Verify document has empty content when 'content' is missing from payload
        # and empty metadata when 'metadata' is missing from payload
        assert len(documents) == 1
        assert documents[0].content == ""
        assert documents[0].meta == {}

    def test_retrieve_with_exception(self, mock_qdrant_pipeline):
        """Test retrieving documents when an exception occurs."""
        query = "test query"
        top_k = 5

        mock_qdrant_pipeline.client.search.side_effect = Exception("Search failed")

        documents = mock_qdrant_pipeline._retrieve(query, top_k)

        # Should return empty list on exception
        assert documents == []
        mock_qdrant_pipeline.logger.error.assert_called_once()
