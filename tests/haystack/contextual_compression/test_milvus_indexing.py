"""Tests for MilvusIndexingPipeline functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.indexing.milvus_indexing import (
    MilvusIndexingPipeline,
)


class TestMilvusIndexingPipeline:
    """Unit tests for MilvusIndexingPipeline functionality."""

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test MilvusIndexingPipeline initialization with default values."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return False (collection doesn't exist)
        mock_utility.has_collection.return_value = False

        pipeline = MilvusIndexingPipeline("config.yaml")

        # Verify connection establishment with defaults
        mock_connections.connect.assert_called_once_with(
            alias="default", host="localhost", port=19530
        )
        # Verify collection creation with defaults
        mock_collection_class.assert_called_once()
        # Check that the call was made with the expected parameters
        call_args = mock_collection_class.call_args
        assert call_args[1]["name"] == "compression"
        assert call_args[1]["using"] == "default"
        # Schema is a complex object created by actual code, just check it exists
        assert "schema" in call_args[1]
        assert pipeline.collection_name == "compression"

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test MilvusIndexingPipeline initialization with custom values."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 512},
            "dataset": {"type": "test", "name": "test_dataset"},
            "milvus": {
                "host": "custom-host",
                "port": 19531,
                "collection_name": "custom_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return False (collection doesn't exist)
        mock_utility.has_collection.return_value = False

        pipeline = MilvusIndexingPipeline("config.yaml")

        # Verify connection establishment with custom values
        mock_connections.connect.assert_called_once_with(
            alias="default", host="custom-host", port=19531
        )
        # Verify collection creation with custom values
        mock_collection_class.assert_called_once()
        # Check that the call was made with the expected parameters
        call_args = mock_collection_class.call_args
        assert call_args[1]["name"] == "custom_collection"
        assert call_args[1]["using"] == "default"
        # Schema is a complex object created by actual code, just check it exists
        assert "schema" in call_args[1]
        assert pipeline.collection_name == "custom_collection"

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test preparing collection creates new collection when it doesn't exist."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "milvus": {
                "collection_name": "new_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return False (collection doesn't exist)
        mock_utility.has_collection.return_value = False

        pipeline = MilvusIndexingPipeline("config.yaml")

        # Verify collection was created (not just accessed)
        mock_collection_class.assert_called_once()
        # Verify create_index was called
        mock_collection_instance.create_index.assert_called_once()
        assert pipeline.collection_name == "new_collection"

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
    def test_prepare_collection_existing_collection_no_drop(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test preparing collection uses existing collection when it exists.

        When drop_existing is False, the existing collection should be used.
        """
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "milvus": {
                "collection_name": "existing_collection",
                "drop_existing": False,
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return True (collection exists)
        mock_utility.has_collection.return_value = True

        pipeline = MilvusIndexingPipeline("config.yaml")

        # Verify collection was NOT created (exists and drop_existing=False)
        mock_collection_class.assert_not_called()
        # Verify drop_collection was NOT called
        mock_utility.drop_collection.assert_not_called()
        assert pipeline.collection_name == "existing_collection"

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
    def test_prepare_collection_existing_collection_with_drop(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test preparing collection drops and recreates existing collection.

        When drop_existing is True, the collection is dropped and recreated.
        """
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "milvus": {
                "collection_name": "to_be_dropped_collection",
                "drop_existing": True,
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return True (collection exists)
        mock_utility.has_collection.return_value = True

        pipeline = MilvusIndexingPipeline("config.yaml")

        # Verify drop_collection was called
        mock_utility.drop_collection.assert_called_once_with("to_be_dropped_collection")
        # Verify collection was created (since it was dropped)
        mock_collection_class.assert_called_once()
        # Verify create_index was called
        mock_collection_instance.create_index.assert_called_once()
        assert pipeline.collection_name == "to_be_dropped_collection"

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test storing documents in Milvus collection."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "milvus": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return False (collection doesn't exist)
        mock_utility.has_collection.return_value = False

        pipeline = MilvusIndexingPipeline("config.yaml")

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

        # Verify collection.insert was called with correct arguments
        mock_collection_instance.insert.assert_called_once()
        call_args = mock_collection_instance.insert.call_args[1]
        data = call_args["data"]

        # Verify the data structure
        assert len(data) == 3  # contents, embeddings, metadata_strs
        contents, embeddings, metadata_strs = data

        # Verify contents
        assert contents == ["Test content 1", "Test content 2"]

        # Verify embeddings
        assert embeddings == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

        # Verify metadata strings
        assert len(metadata_strs) == 2
        assert json.loads(metadata_strs[0]) == {"source": "wiki"}
        assert json.loads(metadata_strs[1]) == {"source": "blog"}

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test storing documents with empty metadata."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "milvus": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return False (collection doesn't exist)
        mock_utility.has_collection.return_value = False

        pipeline = MilvusIndexingPipeline("config.yaml")

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

        # Verify metadata strings are empty JSON objects
        mock_collection_instance.insert.assert_called_once()
        call_args = mock_collection_instance.insert.call_args[1]
        data = call_args["data"]
        _, _, metadata_strs = data

        # Verify metadata strings
        assert len(metadata_strs) == 2
        assert json.loads(metadata_strs[0]) == {}
        assert json.loads(metadata_strs[1]) == {}

    @patch("vectordb.haystack.contextual_compression.indexing.milvus_indexing.utility")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.Collection"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.milvus_indexing.connections"
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
        mock_connections: MagicMock,
        mock_collection_class: MagicMock,
        mock_utility: MagicMock,
    ) -> None:
        """Test storing documents handles failure."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "milvus": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_collection_instance = MagicMock()
        mock_collection_class.return_value = mock_collection_instance

        # Make collection.insert raise an exception
        mock_collection_instance.insert.side_effect = Exception("Milvus insert failed")

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock utility.has_collection to return False (collection doesn't exist)
        mock_utility.has_collection.return_value = False

        pipeline = MilvusIndexingPipeline("config.yaml")

        # Create test documents
        documents = [Document(content="Test content", meta={"source": "wiki"})]
        documents[0].embedding = [0.1, 0.2, 0.3]

        # Verify that the exception is propagated
        with pytest.raises(Exception, match="Milvus insert failed"):
            pipeline._store_documents(documents)
