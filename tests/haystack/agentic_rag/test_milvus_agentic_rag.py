"""Unit tests for MilvusAgenticRAGPipeline class.

Tests all methods and functionality specific to the Milvus implementation.
"""

import json
from unittest.mock import Mock, patch

import pytest
from haystack import Document

from vectordb.haystack.agentic_rag.milvus_agentic_rag import MilvusAgenticRAGPipeline


@pytest.fixture
def mock_milvus_config():
    """Mock configuration for Milvus testing."""
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
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "uri": "http://localhost:19530",
            "token": "test-token",
        },
        "collection": {"name": "test_collection"},
    }


@pytest.fixture
def mock_milvus_pipeline(mock_milvus_config):
    """Create a mock Milvus pipeline instance for testing."""
    # Create a mock pipeline instance without calling the actual constructor
    pipeline = MilvusAgenticRAGPipeline.__new__(
        MilvusAgenticRAGPipeline
    )  # Create without calling __init__

    # Set up the necessary attributes manually
    pipeline.config = mock_milvus_config
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
    pipeline.collection_name = "test_collection"

    return pipeline


class TestMilvusAgenticRAGPipeline:
    """Unit tests for MilvusAgenticRAGPipeline methods."""

    def test_connect_with_token(self, mock_milvus_pipeline):
        """Test connecting to Milvus with token."""
        mock_client = Mock()
        mock_milvus_pipeline.config["milvus"]["token"] = "test-token"

        with patch(
            "vectordb.haystack.agentic_rag.milvus_agentic_rag.MilvusClient",
            return_value=mock_client,
        ) as mock_milvus_client:
            mock_milvus_pipeline._connect()

        mock_milvus_client.assert_called_once_with(
            uri="http://localhost:19530", token="test-token"
        )
        assert mock_milvus_pipeline.client is mock_client
        mock_milvus_pipeline.logger.info.assert_called_once_with(
            "Connected to Milvus at %s", "http://localhost:19530"
        )

    def test_connect_without_token(self, mock_milvus_pipeline):
        """Test connecting to Milvus without token."""
        mock_client = Mock()
        mock_milvus_pipeline.config["milvus"].pop("token", None)

        with patch(
            "vectordb.haystack.agentic_rag.milvus_agentic_rag.MilvusClient",
            return_value=mock_client,
        ) as mock_milvus_client:
            mock_milvus_pipeline._connect()

        mock_milvus_client.assert_called_once_with(uri="http://localhost:19530")
        assert mock_milvus_pipeline.client is mock_client
        mock_milvus_pipeline.logger.info.assert_called_once_with(
            "Connected to Milvus at %s", "http://localhost:19530"
        )

    def test_create_index_existing(self, mock_milvus_pipeline):
        """Test creating Milvus collection when it already exists."""
        mock_stats = {"row_count": 500}
        mock_milvus_pipeline.client.get_collection_stats.return_value = mock_stats
        mock_milvus_pipeline.client.has_collection.return_value = True

        mock_milvus_pipeline._create_index()

        # Verify collection existence was checked
        mock_milvus_pipeline.client.has_collection.assert_called_once_with(
            "test_collection"
        )
        assert mock_milvus_pipeline.collection_name == "test_collection"

    def test_create_index_nonexistent(self, mock_milvus_pipeline):
        """Test creating Milvus collection when it doesn't exist."""
        mock_milvus_pipeline.client.has_collection.return_value = False

        mock_milvus_pipeline._create_index()

        # Should log warning but still set collection name
        assert mock_milvus_pipeline.collection_name == "test_collection"

    def test_create_index_with_exception(self, mock_milvus_pipeline):
        """Test creating Milvus collection when exception occurs."""
        mock_milvus_pipeline.client.has_collection.side_effect = Exception(
            "Check failed"
        )

        mock_milvus_pipeline._create_index()

        # Should log warning but still set collection name
        assert mock_milvus_pipeline.collection_name == "test_collection"

    def test_index_documents_creates_collection_if_needed(self, mock_milvus_pipeline):
        """Test indexing documents creates collection if it doesn't exist."""
        mock_doc1 = Document(
            content="Document 1", embedding=[0.1, 0.2, 0.3], meta={"source": "test1"}
        )
        mock_doc2 = Document(
            content="Document 2", embedding=[0.4, 0.5, 0.6], meta={"source": "test2"}
        )
        mock_milvus_pipeline.embed_documents = Mock(return_value=[mock_doc1, mock_doc2])

        # Mock collection existence check to return False
        mock_milvus_pipeline.client.has_collection.return_value = False

        # Mock the collection creation
        mock_milvus_pipeline.client.create_collection = Mock()

        result = mock_milvus_pipeline.index_documents()

        # Verify collection was created
        mock_milvus_pipeline.client.create_collection.assert_called_once()
        assert result == 2

    def test_index_documents_with_no_documents(self, mock_milvus_pipeline):
        """Test indexing documents when no documents are available."""
        mock_milvus_pipeline.embed_documents = Mock(return_value=[])

        result = mock_milvus_pipeline.index_documents()

        assert result == 0
        mock_milvus_pipeline.logger.warning.assert_called_once_with(
            "No documents to index"
        )

    def test_index_documents_success(self, mock_milvus_pipeline):
        """Test indexing documents successfully."""
        mock_doc1 = Document(
            content="Document 1", embedding=[0.1, 0.2, 0.3], meta={"source": "test1"}
        )
        mock_doc2 = Document(
            content="Document 2", embedding=[0.4, 0.5, 0.6], meta={"source": "test2"}
        )
        mock_milvus_pipeline.embed_documents = Mock(return_value=[mock_doc1, mock_doc2])

        # Mock collection existence check to return True
        mock_milvus_pipeline.client.has_collection.return_value = True

        result = mock_milvus_pipeline.index_documents()

        # Verify documents were inserted into collection
        mock_milvus_pipeline.client.insert.assert_called_once()
        args, kwargs = mock_milvus_pipeline.client.insert.call_args
        assert kwargs["collection_name"] == "test_collection"

        data = kwargs["data"]
        assert len(data) == 3  # content, metadata, embedding
        assert data[0] == ["Document 1", "Document 2"]
        assert data[1] == [
            json.dumps({"source": "test1"}),
            json.dumps({"source": "test2"}),
        ]
        assert data[2] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result == 2

    def test_index_documents_in_batches(self, mock_milvus_pipeline):
        """Test indexing documents in batches."""
        # Create more documents than batch size to trigger batching
        documents = []
        for i in range(150):  # More than batch size of 100
            doc = Document(content=f"Document {i}", embedding=[0.1, 0.2, 0.3])
            documents.append(doc)
        mock_milvus_pipeline.embed_documents = Mock(return_value=documents)

        # Mock collection existence check to return True
        mock_milvus_pipeline.client.has_collection.return_value = True

        result = mock_milvus_pipeline.index_documents()

        # Verify multiple calls to insert for batching
        assert mock_milvus_pipeline.client.insert.call_count == 2  # Two batches
        assert result == 150

    def test_retrieve_success(self, mock_milvus_pipeline):
        """Test retrieving documents successfully."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_milvus_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock search results
        mock_hit1 = {
            "entity": {
                "content": "Content 1",
                "metadata": json.dumps({"field": "value1"}),
            },
            "distance": 0.1,
        }
        mock_hit2 = {
            "entity": {
                "content": "Content 2",
                "metadata": json.dumps({"field": "value2"}),
            },
            "distance": 0.2,
        }

        mock_search_result = [[mock_hit1, mock_hit2]]
        mock_milvus_pipeline.client.search.return_value = mock_search_result

        documents = mock_milvus_pipeline._retrieve(query, top_k)

        # Verify embedding was generated
        mock_milvus_pipeline.dense_embedder.run.assert_called_once_with(text=query)

        # Verify search was called with correct parameters
        mock_milvus_pipeline.client.search.assert_called_once_with(
            collection_name="test_collection",
            data=[mock_embedding],
            limit=top_k,
            output_fields=["content", "metadata"],
        )

        # Verify returned documents
        assert len(documents) == 2
        assert documents[0].content == "Content 1"
        assert documents[0].meta == {"field": "value1"}
        assert documents[0].score == 0.1  # distance value
        assert documents[1].content == "Content 2"
        assert documents[1].meta == {"field": "value2"}
        assert documents[1].score == 0.2  # distance value

    def test_retrieve_with_empty_results(self, mock_milvus_pipeline):
        """Test retrieving documents with empty results."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_milvus_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock empty search results
        mock_milvus_pipeline.client.search.return_value = [[]]

        documents = mock_milvus_pipeline._retrieve(query, top_k)

        assert len(documents) == 0

    def test_retrieve_with_missing_entity(self, mock_milvus_pipeline):
        """Test retrieving documents when entity is missing."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_milvus_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock search result with missing entity
        mock_hit = {
            "distance": 0.1
            # No "entity" key
        }

        mock_search_result = [[mock_hit]]
        mock_milvus_pipeline.client.search.return_value = mock_search_result

        documents = mock_milvus_pipeline._retrieve(query, top_k)

        # Verify document has empty content and metadata when entity is missing
        assert len(documents) == 1
        assert documents[0].content == ""
        assert documents[0].meta == {}

    def test_retrieve_with_invalid_json_metadata(self, mock_milvus_pipeline):
        """Test retrieving documents when metadata is invalid JSON."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_milvus_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock search result with invalid JSON metadata
        mock_hit = {
            "entity": {
                "content": "Content 1",
                "metadata": "{invalid json",  # Invalid JSON
            },
            "distance": 0.1,
        }

        mock_search_result = [[mock_hit]]
        mock_milvus_pipeline.client.search.return_value = mock_search_result

        documents = mock_milvus_pipeline._retrieve(query, top_k)

        # Verify document has empty metadata when JSON is invalid
        assert len(documents) == 1
        assert documents[0].content == "Content 1"
        assert documents[0].meta == {}  # Empty dict due to JSON decode error

    def test_retrieve_with_exception(self, mock_milvus_pipeline):
        """Test retrieving documents when an exception occurs."""
        query = "test query"
        top_k = 5

        mock_milvus_pipeline.client.search.side_effect = Exception("Search failed")

        documents = mock_milvus_pipeline._retrieve(query, top_k)

        # Should return empty list on exception
        assert documents == []
        mock_milvus_pipeline.logger.error.assert_called_once()
