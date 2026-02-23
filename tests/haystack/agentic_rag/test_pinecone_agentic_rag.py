"""Unit tests for PineconeAgenticRAGPipeline class.

Tests all methods and functionality specific to the Pinecone implementation.
"""

from unittest.mock import Mock, patch

import pytest
from haystack import Document

from vectordb.haystack.agentic_rag.pinecone_agentic_rag import (
    PineconeAgenticRAGPipeline,
)


@pytest.fixture
def mock_pinecone_config():
    """Mock configuration for Pinecone testing."""
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
        "pinecone": {
            "api_key": "test-api-key",
        },
        "collection": {"name": "test-index"},
    }


@pytest.fixture
def mock_pinecone_pipeline(mock_pinecone_config):
    """Create a mock Pinecone pipeline instance for testing."""
    # Create a mock pipeline instance without calling the actual constructor
    pipeline = PineconeAgenticRAGPipeline.__new__(
        PineconeAgenticRAGPipeline
    )  # Create without calling __init__

    # Set up the necessary attributes manually
    pipeline.config = mock_pinecone_config
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
    pipeline.index = Mock()
    pipeline.index_name = "test-index"

    return pipeline


class TestPineconeAgenticRAGPipeline:
    """Unit tests for PineconeAgenticRAGPipeline methods."""

    def test_connect_with_api_key_in_config(self, mock_pinecone_pipeline):
        """Test connecting to Pinecone with API key in config."""
        mock_client = Mock()
        mock_pinecone_pipeline.config["pinecone"]["api_key"] = "config-key"

        with patch(
            "vectordb.haystack.agentic_rag.pinecone_agentic_rag.Pinecone",
            return_value=mock_client,
        ) as mock_pinecone:
            mock_pinecone_pipeline._connect()

        mock_pinecone.assert_called_once_with(api_key="config-key")
        assert mock_pinecone_pipeline.client is mock_client
        mock_pinecone_pipeline.logger.info.assert_called_once_with(
            "Connected to Pinecone"
        )

    def test_connect_with_api_key_from_env(self, mock_pinecone_pipeline):
        """Test connecting to Pinecone with API key from environment."""
        mock_client = Mock()
        mock_pinecone_pipeline.config["pinecone"].pop("api_key", None)

        with (
            patch("os.getenv", return_value="env-key"),
            patch(
                "vectordb.haystack.agentic_rag.pinecone_agentic_rag.Pinecone",
                return_value=mock_client,
            ) as mock_pinecone,
        ):
            mock_pinecone_pipeline._connect()

        mock_pinecone.assert_called_once_with(api_key="env-key")
        assert mock_pinecone_pipeline.client is mock_client
        mock_pinecone_pipeline.logger.info.assert_called_once_with(
            "Connected to Pinecone"
        )

    def test_connect_without_api_key(self, mock_pinecone_pipeline):
        """Test connecting to Pinecone without API key raises error."""
        mock_pinecone_pipeline.config["pinecone"].pop("api_key", None)

        with (
            patch("os.getenv", return_value=None),
            pytest.raises(ValueError, match="Pinecone API key is required"),
        ):
            mock_pinecone_pipeline._connect()

    def test_create_index_existing(self, mock_pinecone_pipeline):
        """Test creating Pinecone index when it already exists."""
        mock_index = Mock()
        mock_pinecone_pipeline.client.Index.return_value = mock_index

        # Mock the list_indexes method to return our test index
        mock_index_desc = Mock()
        mock_index_desc.name = "test-index"
        mock_pinecone_pipeline.client.list_indexes.return_value = [mock_index_desc]

        # Mock the describe_index_stats method
        mock_stats = Mock()
        mock_stats.total_vector_count = 500
        mock_index.describe_index_stats.return_value = mock_stats

        mock_pinecone_pipeline._create_index()

        # Verify index was retrieved
        mock_pinecone_pipeline.client.Index.assert_called_once_with("test-index")
        assert mock_pinecone_pipeline.index == mock_index
        assert mock_pinecone_pipeline.index_name == "test-index"

    def test_create_index_nonexistent(self, mock_pinecone_pipeline):
        """Test creating Pinecone index when it doesn't exist."""
        # Mock the list_indexes method to return empty list
        mock_pinecone_pipeline.client.list_indexes.return_value = []

        mock_pinecone_pipeline._create_index()

        # Index should be None since it doesn't exist
        assert mock_pinecone_pipeline.index is None
        assert mock_pinecone_pipeline.index_name == "test-index"

    def test_create_index_with_exception(self, mock_pinecone_pipeline):
        """Test creating Pinecone index when exception occurs during check."""
        mock_pinecone_pipeline.client.list_indexes.side_effect = Exception(
            "List failed"
        )

        mock_pinecone_pipeline._create_index()

        # Index should be None due to exception
        assert mock_pinecone_pipeline.index is None
        assert mock_pinecone_pipeline.index_name == "test-index"

    def test_index_documents_creates_index_if_needed(self, mock_pinecone_pipeline):
        """Test indexing documents creates index if it doesn't exist."""
        mock_doc1 = Document(content="Document 1", embedding=[0.1, 0.2, 0.3])
        mock_doc2 = Document(content="Document 2", embedding=[0.4, 0.5, 0.6])
        mock_pinecone_pipeline.embed_documents = Mock(
            return_value=[mock_doc1, mock_doc2]
        )

        # Initially no index
        mock_pinecone_pipeline.index = None

        # Mock the index creation
        mock_new_index = Mock()
        mock_pinecone_pipeline.client.Index.return_value = mock_new_index
        mock_pinecone_pipeline.client.create_index = Mock()

        result = mock_pinecone_pipeline.index_documents()

        # Verify index was created
        mock_pinecone_pipeline.client.create_index.assert_called_once_with(
            name="test-index",
            dimension=3,  # Length of embedding
            metric="cosine",
        )
        assert result == 2

    def test_index_documents_with_no_documents(self, mock_pinecone_pipeline):
        """Test indexing documents when no documents are available."""
        mock_pinecone_pipeline.embed_documents = Mock(return_value=[])

        result = mock_pinecone_pipeline.index_documents()

        assert result == 0
        mock_pinecone_pipeline.logger.warning.assert_called_once_with(
            "No documents to index"
        )

    def test_index_documents_success(self, mock_pinecone_pipeline):
        """Test indexing documents successfully."""
        mock_doc1 = Document(content="Document 1", embedding=[0.1, 0.2, 0.3])
        mock_doc2 = Document(content="Document 2", embedding=[0.4, 0.5, 0.6])
        mock_pinecone_pipeline.embed_documents = Mock(
            return_value=[mock_doc1, mock_doc2]
        )

        mock_index = Mock()
        mock_pinecone_pipeline.index = mock_index

        result = mock_pinecone_pipeline.index_documents()

        # Verify documents were upserted to index
        mock_index.upsert.assert_called_once()
        args, kwargs = mock_index.upsert.call_args
        vectors = kwargs["vectors"]

        # Verify the vectors were constructed correctly
        assert len(vectors) == 2
        assert vectors[0] == (
            "0",
            [0.1, 0.2, 0.3],
            {"content": "Document 1", "metadata": {}},
        )
        assert vectors[1] == (
            "1",
            [0.4, 0.5, 0.6],
            {"content": "Document 2", "metadata": {}},
        )
        assert result == 2

    def test_index_documents_in_batches(self, mock_pinecone_pipeline):
        """Test indexing documents in batches."""
        # Create more documents than batch size to trigger batching
        documents = []
        for i in range(150):  # More than batch size of 100
            doc = Document(content=f"Document {i}", embedding=[0.1, 0.2, 0.3])
            documents.append(doc)
        mock_pinecone_pipeline.embed_documents = Mock(return_value=documents)

        mock_index = Mock()
        mock_pinecone_pipeline.index = mock_index

        result = mock_pinecone_pipeline.index_documents()

        # Verify multiple calls to upsert for batching
        assert mock_index.upsert.call_count == 2  # Two batches
        assert result == 150

    def test_retrieve_success(self, mock_pinecone_pipeline):
        """Test retrieving documents successfully."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_pinecone_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Mock the index
        mock_index = Mock()
        mock_pinecone_pipeline.index = mock_index

        # Mock query results
        mock_match1 = Mock()
        mock_match1.score = 0.9
        mock_match1.metadata = {"content": "Content 1", "field": "value1"}
        mock_match2 = Mock()
        mock_match2.score = 0.8
        mock_match2.metadata = {"content": "Content 2", "field": "value2"}

        mock_query_result = Mock()
        mock_query_result.matches = [mock_match1, mock_match2]
        mock_index.query.return_value = mock_query_result

        documents = mock_pinecone_pipeline._retrieve(query, top_k)

        # Verify embedding was generated
        mock_pinecone_pipeline.dense_embedder.run.assert_called_once_with(text=query)

        # Verify query was called with correct parameters
        mock_index.query.assert_called_once_with(
            vector=mock_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        # Verify returned documents
        assert len(documents) == 2
        assert documents[0].content == "Content 1"
        assert documents[0].meta == {"field": "value1"}
        assert documents[0].score == 0.9
        assert documents[1].content == "Content 2"
        assert documents[1].meta == {"field": "value2"}
        assert documents[1].score == 0.8

    def test_retrieve_with_no_index_yet(self, mock_pinecone_pipeline):
        """Test retrieving documents when no index exists yet."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_pinecone_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Initially no index
        mock_pinecone_pipeline.index = None

        # Mock the index retrieval
        mock_index = Mock()
        mock_pinecone_pipeline.client.Index.return_value = mock_index

        # Mock query results
        mock_query_result = Mock()
        mock_query_result.matches = []
        mock_index.query.return_value = mock_query_result

        mock_pinecone_pipeline._retrieve(query, top_k)

        # Verify index was retrieved from client
        mock_pinecone_pipeline.client.Index.assert_called_once_with("test-index")
        assert mock_pinecone_pipeline.index == mock_index

    def test_retrieve_with_empty_matches(self, mock_pinecone_pipeline):
        """Test retrieving documents with empty matches."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_pinecone_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        mock_index = Mock()
        mock_pinecone_pipeline.index = mock_index

        # Mock empty query results
        mock_query_result = Mock()
        mock_query_result.matches = []
        mock_index.query.return_value = mock_query_result

        documents = mock_pinecone_pipeline._retrieve(query, top_k)

        assert len(documents) == 0

    def test_retrieve_with_missing_content_in_metadata(self, mock_pinecone_pipeline):
        """Test retrieving documents when content is missing from metadata."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_pinecone_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        mock_index = Mock()
        mock_pinecone_pipeline.index = mock_index

        # Mock query result with missing content in metadata
        mock_match = Mock()
        mock_match.score = 0.9
        mock_match.metadata = {"field": "value"}  # No 'content' key
        mock_query_result = Mock()
        mock_query_result.matches = [mock_match]
        mock_index.query.return_value = mock_query_result

        documents = mock_pinecone_pipeline._retrieve(query, top_k)

        # Verify document has empty content when 'content' is missing from metadata
        assert len(documents) == 1
        assert documents[0].content == ""
        assert documents[0].meta == {"field": "value"}

    def test_retrieve_with_exception(self, mock_pinecone_pipeline):
        """Test retrieving documents when an exception occurs."""
        query = "test query"
        top_k = 5

        mock_index = Mock()
        mock_pinecone_pipeline.index = mock_index
        mock_index.query.side_effect = Exception("Query failed")

        documents = mock_pinecone_pipeline._retrieve(query, top_k)

        # Should return empty list on exception
        assert documents == []
        mock_pinecone_pipeline.logger.error.assert_called_once()
