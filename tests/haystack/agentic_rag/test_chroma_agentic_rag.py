"""Unit tests for ChromaAgenticRAGPipeline class.

Tests all methods and functionality specific to the Chroma implementation.
"""

from unittest.mock import Mock, patch

import pytest
from haystack import Document

from vectordb.haystack.agentic_rag.chroma_agentic_rag import ChromaAgenticRAGPipeline


@pytest.fixture
def mock_chroma_config():
    """Mock configuration for Chroma testing."""
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
        "chroma": {
            "host": "localhost",
            "port": 8000,
            "persist_directory": None,
        },
        "collection": {"name": "test-collection"},
    }


@pytest.fixture
def mock_chroma_pipeline(mock_chroma_config):
    """Create a mock Chroma pipeline instance for testing."""
    # Create a mock pipeline instance without calling the actual constructor
    pipeline = ChromaAgenticRAGPipeline.__new__(
        ChromaAgenticRAGPipeline
    )  # Create without calling __init__

    # Set up the necessary attributes manually
    pipeline.config = mock_chroma_config
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
    pipeline.collection = Mock()
    pipeline.collection_name = "test-collection"

    return pipeline


class TestChromaAgenticRAGPipeline:
    """Unit tests for ChromaAgenticRAGPipeline methods."""

    def test_connect_with_persistence(self, mock_chroma_pipeline):
        """Test connecting to Chroma with persistence."""
        mock_chroma_pipeline.config["chroma"]["persist_directory"] = "/tmp/chroma"
        mock_chromadb = Mock()
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            mock_chroma_pipeline._connect()

        mock_chromadb.PersistentClient.assert_called_once_with(path="/tmp/chroma")
        assert mock_chroma_pipeline.client is mock_client
        mock_chroma_pipeline.logger.info.assert_called_once_with(
            "Connected to Chroma with persistence at %s", "/tmp/chroma"
        )

    def test_connect_with_http_client(self, mock_chroma_pipeline):
        """Test connecting to Chroma with HTTP client."""
        mock_chromadb = Mock()
        mock_client = Mock()
        mock_chromadb.HttpClient.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            mock_chroma_pipeline._connect()

        mock_chromadb.HttpClient.assert_called_once_with(host="localhost", port=8000)
        assert mock_chroma_pipeline.client is mock_client
        mock_chroma_pipeline.logger.info.assert_called_once_with(
            "Connected to Chroma at %s:%s", "localhost", 8000
        )

    def test_connect_with_ephemeral_client(self, mock_chroma_pipeline):
        """Test connecting to Chroma with ephemeral client (when HTTP fails)."""
        mock_chromadb = Mock()
        mock_chromadb.HttpClient.side_effect = Exception("http failed")
        mock_client = Mock()
        mock_chromadb.Client.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            mock_chroma_pipeline._connect()

        mock_chromadb.HttpClient.assert_called_once_with(host="localhost", port=8000)
        mock_chromadb.Client.assert_called_once_with()
        assert mock_chroma_pipeline.client is mock_client
        mock_chroma_pipeline.logger.info.assert_called_once_with(
            "Connected to ephemeral Chroma client"
        )

    def test_create_index_success(self, mock_chroma_pipeline):
        """Test creating Chroma collection successfully."""
        mock_collection = Mock()
        mock_collection.count.return_value = 100
        mock_chroma_pipeline.client.get_or_create_collection.return_value = (
            mock_collection
        )

        mock_chroma_pipeline._create_index()

        # Verify collection was created/get with correct parameters
        mock_chroma_pipeline.client.get_or_create_collection.assert_called_once_with(
            name="test-collection",
            metadata={"hnsw:space": "cosine"},
        )
        assert mock_chroma_pipeline.collection == mock_collection
        assert mock_chroma_pipeline.collection_name == "test-collection"

    def test_create_index_failure(self, mock_chroma_pipeline):
        """Test creating Chroma collection with failure."""
        mock_chroma_pipeline.client.get_or_create_collection.side_effect = RuntimeError(
            "Creation failed"
        )

        with pytest.raises(RuntimeError):
            mock_chroma_pipeline._create_index()

    def test_index_documents_with_no_documents(self, mock_chroma_pipeline):
        """Test indexing documents when no documents are available."""
        mock_chroma_pipeline.embed_documents = Mock(return_value=[])

        result = mock_chroma_pipeline.index_documents()

        assert result == 0
        mock_chroma_pipeline.logger.warning.assert_called_once_with(
            "No documents to index"
        )

    def test_retrieve_handles_empty_documents_key(self, mock_chroma_pipeline):
        """Test retrieving documents when documents key is empty."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_chroma_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        mock_chroma_pipeline.collection.query.return_value = {
            "documents": [],
            "metadatas": [],
            "distances": [],
        }

        documents = mock_chroma_pipeline._retrieve(query, top_k)

        assert documents == []

    def test_index_documents_success(self, mock_chroma_pipeline):
        """Test indexing documents successfully."""
        mock_doc1 = Document(content="Document 1", embedding=[0.1, 0.2, 0.3])
        mock_doc2 = Document(content="Document 2", embedding=[0.4, 0.5, 0.6])
        mock_chroma_pipeline.embed_documents = Mock(return_value=[mock_doc1, mock_doc2])

        result = mock_chroma_pipeline.index_documents()

        # Verify documents were added to collection
        mock_chroma_pipeline.collection.add.assert_called_once()
        args, kwargs = mock_chroma_pipeline.collection.add.call_args
        assert kwargs["ids"] == ["0", "1"]
        assert kwargs["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert kwargs["documents"] == ["Document 1", "Document 2"]
        assert kwargs["metadatas"] == [{}, {}]
        assert result == 2

    def test_index_documents_in_batches(self, mock_chroma_pipeline):
        """Test indexing documents in batches."""
        # Create more documents than batch size to trigger batching
        documents = []
        for i in range(150):  # More than batch size of 100
            doc = Document(content=f"Document {i}", embedding=[0.1, 0.2, 0.3])
            documents.append(doc)
        mock_chroma_pipeline.embed_documents = Mock(return_value=documents)

        result = mock_chroma_pipeline.index_documents()

        # Verify multiple calls to add for batching
        assert mock_chroma_pipeline.collection.add.call_count == 2  # Two batches
        assert result == 150

    def test_retrieve_success(self, mock_chroma_pipeline):
        """Test retrieving documents successfully."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_chroma_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        mock_results = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.2]],
        }
        mock_chroma_pipeline.collection.query.return_value = mock_results

        documents = mock_chroma_pipeline._retrieve(query, top_k)

        # Verify embedding was generated
        mock_chroma_pipeline.dense_embedder.run.assert_called_once_with(text=query)

        # Verify query was called with correct parameters
        mock_chroma_pipeline.collection.query.assert_called_once_with(
            query_embeddings=[mock_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Verify returned documents
        assert len(documents) == 2
        assert documents[0].content == "Doc 1"
        assert documents[0].meta == {"source": "test1"}
        assert documents[0].score == 0.9  # 1.0 - 0.1
        assert documents[1].content == "Doc 2"
        assert documents[1].meta == {"source": "test2"}
        assert documents[1].score == 0.8  # 1.0 - 0.2

    def test_retrieve_with_empty_results(self, mock_chroma_pipeline):
        """Test retrieving documents with empty results."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_chroma_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Return empty results
        mock_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_chroma_pipeline.collection.query.return_value = mock_results

        documents = mock_chroma_pipeline._retrieve(query, top_k)

        assert len(documents) == 0

    def test_retrieve_with_missing_metadatas(self, mock_chroma_pipeline):
        """Test retrieving documents when metadatas are missing."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_chroma_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Return results without metadatas
        mock_results = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": None,
            "distances": [[0.1, 0.2]],
        }
        mock_chroma_pipeline.collection.query.return_value = mock_results

        documents = mock_chroma_pipeline._retrieve(query, top_k)

        # Verify documents have empty metadata when metadatas is None
        assert len(documents) == 2
        assert documents[0].meta == {}
        assert documents[1].meta == {}

    def test_retrieve_with_missing_distances(self, mock_chroma_pipeline):
        """Test retrieving documents when distances are missing."""
        query = "test query"
        top_k = 5

        mock_embedding = [0.1, 0.2, 0.3]
        mock_chroma_pipeline.dense_embedder.run.return_value = {
            "embedding": mock_embedding
        }

        # Return results without distances - in the actual implementation,
        # when distances is None, it defaults to [0.0] * len(docs_list),
        # so score becomes 1.0 - 0.0 = 1.0
        mock_results = {
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": None,  # This triggers the default [0.0, 0.0] in actual code
        }
        mock_chroma_pipeline.collection.query.return_value = mock_results

        documents = mock_chroma_pipeline._retrieve(query, top_k)

        # Verify documents have score of 1.0 when distances defaults to [0.0]
        assert len(documents) == 2
        assert documents[0].score == 1.0
        assert documents[1].score == 1.0

    def test_retrieve_with_exception(self, mock_chroma_pipeline):
        """Test retrieving documents when an exception occurs."""
        query = "test query"
        top_k = 5

        mock_chroma_pipeline.collection.query.side_effect = Exception("Query failed")

        documents = mock_chroma_pipeline._retrieve(query, top_k)

        # Should return empty list on exception
        assert documents == []
        mock_chroma_pipeline.logger.error.assert_called_once()
