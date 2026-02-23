"""Tests for PineconeIndexingPipeline functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.indexing.pinecone_indexing import (
    PineconeIndexingPipeline,
)


class TestPineconeIndexingPipeline:
    """Unit tests for PineconeIndexingPipeline functionality."""

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_initialization_with_api_key(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test PineconeIndexingPipeline initialization with API key."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-index",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index
        mock_pinecone_instance.list_indexes.return_value = []

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = PineconeIndexingPipeline("config.yaml")

        # Verify Pinecone client initialization
        mock_pinecone_class.assert_called_once_with(api_key="test-api-key")
        assert pipeline.pc == mock_pinecone_instance
        assert pipeline.index == mock_index

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_initialization_without_api_key_raises_error(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test PineconeIndexingPipeline initialization without API key raises error."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                # Missing api_key
                "index_name": "test-index",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        with pytest.raises(
            ValueError, match="pinecone.api_key not specified in config"
        ):
            PineconeIndexingPipeline("config.yaml")

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_prepare_collection_new_index(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test preparing collection creates new index when it doesn't exist."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "new-index",
                "metric": "cosine",
                "cloud": "aws",
                "region": "us-west-2",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        # Simulate that index doesn't exist
        mock_pinecone_instance.list_indexes.return_value = []

        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = PineconeIndexingPipeline("config.yaml")

        # Verify that create_index was called
        mock_pinecone_instance.create_index.assert_called_once()
        # Check that the call was made with the expected parameters
        call_args = mock_pinecone_instance.create_index.call_args
        assert call_args[1]["name"] == "new-index"
        assert call_args[1]["dimension"] == 384
        assert call_args[1]["metric"] == "cosine"
        # The spec parameter is a ServerlessSpec object, which we can't easily mock
        # So we just check that it was passed
        assert "spec" in call_args[1]
        assert pipeline.index == mock_index

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_prepare_collection_existing_index(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test preparing collection uses existing index when it exists."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "existing-index",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        # Simulate that index exists
        mock_existing_index = MagicMock()
        mock_existing_index.name = "existing-index"
        mock_pinecone_instance.list_indexes.return_value = [mock_existing_index]

        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = PineconeIndexingPipeline("config.yaml")

        # Verify that create_index was NOT called
        mock_pinecone_instance.create_index.assert_not_called()
        # Verify that Index was called to get the existing index
        mock_pinecone_instance.Index.assert_called_once_with("existing-index")
        assert pipeline.index == mock_index

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_store_documents_success(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test storing documents in Pinecone index."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-index",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_pinecone_instance.list_indexes.return_value = []

        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = PineconeIndexingPipeline("config.yaml")

        # Create test documents
        documents = [
            Document(content="Test content 1", meta={"source": "wiki"}),
            Document(content="Test content 2", meta={"source": "blog"}),
        ]

        # Set embeddings for the documents
        for doc in documents:
            doc.embedding = [0.1, 0.2, 0.3]

        # Call _store_documents
        pipeline._store_documents(documents)

        # Verify index.upsert was called with correct arguments
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args[1]
        vectors = call_args["vectors"]

        # Verify the number of vectors
        assert len(vectors) == 2

        # Verify each vector has the correct structure
        for i, vector in enumerate(vectors):
            assert "id" in vector
            assert "values" in vector
            assert "metadata" in vector
            assert vector["values"] == [0.1, 0.2, 0.3]

            # Verify metadata structure
            metadata = vector["metadata"]
            assert "content" in metadata
            assert "metadata_json" in metadata
            assert metadata["content"] == f"Test content {i + 1}"

            # Verify metadata_json is properly serialized
            expected_meta = json.loads(metadata["metadata_json"])
            assert expected_meta == {"source": ["wiki", "blog"][i]}

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_store_documents_with_long_content(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test storing documents with content that exceeds Pinecone limits."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-index",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_pinecone_instance.list_indexes.return_value = []

        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = PineconeIndexingPipeline("config.yaml")

        # Create a document with very long content (longer than 50000 chars limit)
        long_content = "This is a very long content. " * 2000  # More than 50000 chars
        documents = [Document(content=long_content, meta={"source": "wiki"})]
        documents[0].embedding = [0.1, 0.2, 0.3]

        # Call _store_documents
        pipeline._store_documents(documents)

        # Verify index.upsert was called with truncated content
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args[1]
        vectors = call_args["vectors"]

        # Verify content was truncated
        metadata = vectors[0]["metadata"]
        assert len(metadata["content"]) <= 50000
        assert metadata["content"] == long_content[:50000]

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_store_documents_with_empty_metadata(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test storing documents with empty metadata."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-index",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_pinecone_instance.list_indexes.return_value = []

        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = PineconeIndexingPipeline("config.yaml")

        # Create test documents with no metadata
        documents = [
            Document(content="Test content 1"),
            Document(content="Test content 2", meta={}),
        ]

        # Set embeddings for the documents
        for doc in documents:
            doc.embedding = [0.1, 0.2, 0.3]

        # Call _store_documents
        pipeline._store_documents(documents)

        # Verify index.upsert was called with correct metadata
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args[1]
        vectors = call_args["vectors"]

        # Verify metadata_json is properly serialized for empty metadata
        for i, vector in enumerate(vectors):
            metadata = vector["metadata"]
            expected_meta = json.loads(metadata["metadata_json"])
            if i == 0:
                # Document without meta should have empty dict
                assert expected_meta == {}
            else:
                # Document with empty meta dict should have empty dict
                assert expected_meta == {}

    @patch(
        "vectordb.haystack.contextual_compression.indexing.pinecone_indexing.Pinecone"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_store_documents_failure(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test storing documents handles failure."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-index",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_pinecone_instance = MagicMock()
        mock_pinecone_class.return_value = mock_pinecone_instance

        mock_pinecone_instance.list_indexes.return_value = []

        mock_index = MagicMock()
        mock_pinecone_instance.Index.return_value = mock_index

        # Make index.upsert raise an exception
        mock_index.upsert.side_effect = Exception("Pinecone upsert failed")

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = PineconeIndexingPipeline("config.yaml")

        # Create test documents
        documents = [Document(content="Test content", meta={"source": "wiki"})]
        documents[0].embedding = [0.1, 0.2, 0.3]

        # Verify that the exception is propagated
        with pytest.raises(Exception, match="Pinecone upsert failed"):
            pipeline._store_documents(documents)
