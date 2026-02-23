"""Tests for WeaviateIndexingPipeline functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.indexing.weaviate_indexing import (
    WeaviateIndexingPipeline,
)


class TestWeaviateIndexingPipeline:
    """Unit tests for WeaviateIndexingPipeline functionality."""

    @patch(
        "vectordb.haystack.contextual_compression.indexing.weaviate_indexing.weaviate"
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
    def test_initialization_with_defaults(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_weaviate_module: MagicMock,
    ) -> None:
        """Test WeaviateIndexingPipeline initialization with default values."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = WeaviateIndexingPipeline("config.yaml")

        # Verify Weaviate client initialization with defaults
        mock_weaviate_module.connect_to_local.assert_called_once_with(
            url="http://localhost:8080"
        )
        assert pipeline.client == mock_client
        assert pipeline.collection == mock_collection

    @patch(
        "vectordb.haystack.contextual_compression.indexing.weaviate_indexing.weaviate"
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
    def test_initialization_with_remote_url(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_weaviate_module: MagicMock,
    ) -> None:
        """Test WeaviateIndexingPipeline initialization with remote URL and API key."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "weaviate": {
                "url": "https://my-cluster.weaviate.network",
                "api_key": "my-api-key",
                "collection_name": "custom_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_client.collections.exists.return_value = True
        mock_weaviate_module.connect_to_cloud.return_value = mock_client

        mock_auth_api_key = MagicMock()
        mock_weaviate_module.auth.AuthApiKey.return_value = mock_auth_api_key

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = WeaviateIndexingPipeline("config.yaml")

        # Verify connect_to_cloud was called with correct parameters
        mock_weaviate_module.connect_to_cloud.assert_called_once_with(
            cluster_url="https://my-cluster.weaviate.network",
            auth_credentials=mock_auth_api_key,
        )
        mock_weaviate_module.auth.AuthApiKey.assert_called_once_with("my-api-key")
        assert pipeline.client == mock_client
        assert pipeline.collection == mock_collection

    @patch(
        "vectordb.haystack.contextual_compression.indexing.weaviate_indexing.weaviate"
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
    def test_prepare_collection_new_collection(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_weaviate_module: MagicMock,
    ) -> None:
        """Test preparing collection creates new collection when it doesn't exist."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "weaviate": {
                "collection_name": "new_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.exists.return_value = False  # Collection doesn't exist
        mock_client.collections.create.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_configure = MagicMock()
        mock_weaviate_module.classes.config.Configure = mock_configure

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = WeaviateIndexingPipeline("config.yaml")

        # Verify collections.create was called
        mock_client.collections.create.assert_called_once()
        call_args = mock_client.collections.create.call_args
        assert call_args[1]["name"] == "new_collection"
        assert pipeline.collection == mock_collection

    @patch(
        "vectordb.haystack.contextual_compression.indexing.weaviate_indexing.weaviate"
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
    def test_prepare_collection_existing_collection(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_weaviate_module: MagicMock,
    ) -> None:
        """Test preparing collection uses existing collection when it exists."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "weaviate": {
                "collection_name": "existing_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.exists.return_value = True  # Collection exists
        mock_client.collections.get.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = WeaviateIndexingPipeline("config.yaml")

        # Verify collections.get was called (not create)
        mock_client.collections.get.assert_called_once_with("existing_collection")
        # Verify collections.create was NOT called
        mock_client.collections.create.assert_not_called()
        assert pipeline.collection == mock_collection

    @patch(
        "vectordb.haystack.contextual_compression.indexing.weaviate_indexing.weaviate"
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
        mock_weaviate_module: MagicMock,
    ) -> None:
        """Test storing documents in Weaviate collection."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "weaviate": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_batch_context = MagicMock()
        # Properly mock the context manager behavior
        mock_batch_context.__enter__ = MagicMock(return_value=mock_batch_context)
        mock_batch_context.__exit__ = MagicMock(return_value=None)
        mock_collection.batch.dynamic.return_value = mock_batch_context
        mock_client.collections.exists.return_value = False  # Collection doesn't exist
        mock_client.collections.create.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_configure = MagicMock()
        mock_weaviate_module.classes.config.Configure = mock_configure

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = WeaviateIndexingPipeline("config.yaml")

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

        # Verify batch.add_object was called for each document
        assert mock_batch_context.add_object.call_count == 2

        # Verify the calls were made with correct arguments
        calls = mock_batch_context.add_object.call_args_list
        for i, call in enumerate(calls):
            args, kwargs = call
            # Verify properties
            assert kwargs["properties"]["content"] == f"Test content {i + 1}"
            # Verify metadata_json is properly serialized
            expected_meta = json.loads(kwargs["properties"]["metadata_json"])
            assert expected_meta == {"source": ["wiki", "blog"][i]}
            # Verify vector
            assert kwargs["vector"] == [0.1, 0.2, 0.3]

    @patch(
        "vectordb.haystack.contextual_compression.indexing.weaviate_indexing.weaviate"
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
        mock_weaviate_module: MagicMock,
    ) -> None:
        """Test storing documents with empty metadata."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "weaviate": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_batch_context = MagicMock()
        # Properly mock the context manager behavior
        mock_batch_context.__enter__ = MagicMock(return_value=mock_batch_context)
        mock_batch_context.__exit__ = MagicMock(return_value=None)
        mock_collection.batch.dynamic.return_value = mock_batch_context
        mock_client.collections.exists.return_value = False  # Collection doesn't exist
        mock_client.collections.create.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_configure = MagicMock()
        mock_weaviate_module.classes.config.Configure = mock_configure

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = WeaviateIndexingPipeline("config.yaml")

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
        calls = mock_batch_context.add_object.call_args_list
        for i, call in enumerate(calls):
            args, kwargs = call
            # Verify metadata_json for each document
            expected_meta = json.loads(kwargs["properties"]["metadata_json"])
            if i == 0:
                # Document without meta should have empty dict
                assert expected_meta == {}
            else:
                # Document with empty meta dict should have empty dict
                assert expected_meta == {}

    @patch(
        "vectordb.haystack.contextual_compression.indexing.weaviate_indexing.weaviate"
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
        mock_weaviate_module: MagicMock,
    ) -> None:
        """Test storing documents handles failure."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2", "dimension": 384},
            "dataset": {"type": "test", "name": "test_dataset"},
            "weaviate": {
                "collection_name": "test_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_batch_context = MagicMock()
        mock_collection.batch.dynamic.return_value = mock_batch_context
        mock_client.collections.exists.return_value = False  # Collection doesn't exist
        mock_client.collections.create.return_value = mock_collection
        mock_weaviate_module.connect_to_local.return_value = mock_client

        mock_configure = MagicMock()
        mock_weaviate_module.classes.config.Configure = mock_configure

        # Make batch.add_object raise an exception
        mock_batch_context.__enter__ = MagicMock(
            side_effect=Exception("Weaviate batch failed")
        )

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = WeaviateIndexingPipeline("config.yaml")

        # Create test documents
        documents = [Document(content="Test content", meta={"source": "wiki"})]
        documents[0].embedding = [0.1, 0.2, 0.3]

        # Verify that the exception is propagated
        with pytest.raises(Exception, match="Weaviate batch failed"):
            pipeline._store_documents(documents)
