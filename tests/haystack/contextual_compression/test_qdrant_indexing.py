"""Tests for QdrantIndexingPipeline functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.indexing.qdrant_indexing import (
    QdrantIndexingPipeline,
)


class TestQdrantIndexingPipeline:
    """Unit tests for QdrantIndexingPipeline functionality."""

    @patch(
        "vectordb.haystack.contextual_compression.indexing.qdrant_indexing.QdrantClient"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_initialization_with_defaults(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_qdrant_client_class: MagicMock,
    ) -> None:
        """Test QdrantIndexingPipeline initialization with default values."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_qdrant_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_qdrant_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock get_collection to raise an exception (collection doesn't exist)
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        pipeline = QdrantIndexingPipeline("config.yaml")

        # Verify Qdrant client initialization with defaults
        mock_qdrant_client_class.assert_called_once_with(url="http://localhost:6333")
        assert pipeline.client == mock_qdrant_client
        assert pipeline.collection_name == "compression"

    @patch(
        "vectordb.haystack.contextual_compression.indexing.qdrant_indexing.QdrantClient"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_initialization_with_custom_values(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_qdrant_client_class: MagicMock,
    ) -> None:
        """Test QdrantIndexingPipeline initialization with custom values."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 512},
            "dataset": {"type": "test", "name": "test_dataset"},
            "qdrant": {
                "url": "http://custom-qdrant:6333",
                "api_key": "custom-api-key",
                "collection_name": "custom_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_qdrant_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_qdrant_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock get_collection to raise an exception (collection doesn't exist)
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        pipeline = QdrantIndexingPipeline("config.yaml")

        # Verify Qdrant client initialization with custom values
        mock_qdrant_client_class.assert_called_once_with(
            url="http://custom-qdrant:6333", api_key="custom-api-key"
        )
        assert pipeline.client == mock_qdrant_client
        assert pipeline.collection_name == "custom_collection"

    @patch(
        "vectordb.haystack.contextual_compression.indexing.qdrant_indexing.QdrantClient"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_prepare_collection_new_collection(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_qdrant_client_class: MagicMock,
    ) -> None:
        """Test preparing collection creates new collection when it doesn't exist."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "qdrant": {
                "collection_name": "new_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_qdrant_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_qdrant_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock get_collection to raise an exception (collection doesn't exist)
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        QdrantIndexingPipeline("config.yaml")

        # Verify create_collection was called
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "new_collection"
        assert call_args[1]["vectors_config"].size == 384

    @patch(
        "vectordb.haystack.contextual_compression.indexing.qdrant_indexing.QdrantClient"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_prepare_collection_existing_collection(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_qdrant_client_class: MagicMock,
    ) -> None:
        """Test preparing collection uses existing collection when it exists."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "qdrant": {
                "collection_name": "existing_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_qdrant_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_qdrant_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock get_collection to return successfully (collection exists)
        mock_qdrant_client.get_collection.return_value = MagicMock()

        pipeline = QdrantIndexingPipeline("config.yaml")

        # Verify get_collection was called
        mock_qdrant_client.get_collection.assert_called_once_with("existing_collection")
        # Verify create_collection was NOT called
        mock_qdrant_client.create_collection.assert_not_called()
        assert pipeline.collection_name == "existing_collection"

    @patch(
        "vectordb.haystack.contextual_compression.indexing.qdrant_indexing.QdrantClient"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_store_documents_success(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_qdrant_client_class: MagicMock,
    ) -> None:
        """Test storing documents in Qdrant collection."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "qdrant": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_qdrant_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_qdrant_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock get_collection to raise an exception (collection doesn't exist)
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        pipeline = QdrantIndexingPipeline("config.yaml")

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

        # Verify client.upsert was called with correct arguments
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args[1]

        assert call_args["collection_name"] == "test_collection"
        points = call_args["points"]

        # Verify the number of points
        assert len(points) == 2

        # Verify each point has the correct structure
        for i, point in enumerate(points):
            assert hasattr(point, "id")
            assert point.vector == [0.1, 0.2, 0.3]
            assert point.payload["content"] == f"Test content {i + 1}"

            # Verify metadata_json is properly serialized
            expected_meta = json.loads(point.payload["metadata_json"])
            assert expected_meta == {"source": ["wiki", "blog"][i]}

    @patch(
        "vectordb.haystack.contextual_compression.indexing.qdrant_indexing.QdrantClient"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_store_documents_with_empty_metadata(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_qdrant_client_class: MagicMock,
    ) -> None:
        """Test storing documents with empty metadata."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "qdrant": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_qdrant_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_qdrant_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock get_collection to raise an exception (collection doesn't exist)
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        pipeline = QdrantIndexingPipeline("config.yaml")

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

        # Verify metadata_json is properly serialized for empty metadata
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args[1]
        points = call_args["points"]

        # Verify metadata_json for each point
        for i, point in enumerate(points):
            expected_meta = json.loads(point.payload["metadata_json"])
            if i == 0:
                # Document without meta should have empty dict
                assert expected_meta == {}
            else:
                # Document with empty meta dict should have empty dict
                assert expected_meta == {}

    @patch(
        "vectordb.haystack.contextual_compression.indexing.qdrant_indexing.QdrantClient"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_store_documents_failure(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_qdrant_client_class: MagicMock,
    ) -> None:
        """Test storing documents handles failure."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "qdrant": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_qdrant_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Make client.upsert raise an exception
        mock_qdrant_client.upsert.side_effect = Exception("Qdrant upsert failed")

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock get_collection to raise an exception (collection doesn't exist)
        mock_qdrant_client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        pipeline = QdrantIndexingPipeline("config.yaml")

        # Create test documents
        documents = [Document(content="Test content", meta={"source": "wiki"})]
        documents[0].embedding = [0.1, 0.2, 0.3]

        # Verify that the exception is propagated
        with pytest.raises(Exception, match="Qdrant upsert failed"):
            pipeline._store_documents(documents)
