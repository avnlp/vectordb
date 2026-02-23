"""Tests for ChromaIndexingPipeline functionality."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.indexing.chroma_indexing import (
    ChromaIndexingPipeline,
)


class TestChromaIndexingPipeline:
    """Unit tests for ChromaIndexingPipeline functionality."""

    @patch("vectordb.haystack.contextual_compression.indexing.chroma_indexing.chromadb")
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
        mock_chromadb_module: MagicMock,
    ) -> None:
        """Test ChromaIndexingPipeline initialization with default values."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_chromadb_module.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ChromaIndexingPipeline("config.yaml")

        # Verify client initialization with default path
        mock_chromadb_module.PersistentClient.assert_called_once_with(
            path="./chroma_data"
        )
        # Verify collection creation with default name
        mock_client.get_or_create_collection.assert_called_once_with(
            name="compression",
            metadata={"hnsw:space": "cosine"},
        )
        assert pipeline.client == mock_client
        assert pipeline.collection == mock_collection

    @patch("vectordb.haystack.contextual_compression.indexing.chroma_indexing.chromadb")
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
        mock_chromadb_module: MagicMock,
    ) -> None:
        """Test ChromaIndexingPipeline initialization with custom values."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
            "chroma": {
                "path": "/custom/path",
                "persist_directory": "/custom/persist",
                "collection_name": "custom_collection",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_chromadb_module.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        ChromaIndexingPipeline("config.yaml")

        # Verify client initialization with custom path
        mock_chromadb_module.PersistentClient.assert_called_once_with(
            path="/custom/persist"
        )
        # Verify collection creation with custom name
        mock_client.get_or_create_collection.assert_called_once_with(
            name="custom_collection",
            metadata={"hnsw:space": "cosine"},
        )

    @patch("vectordb.haystack.contextual_compression.indexing.chroma_indexing.chromadb")
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
        mock_chromadb_module: MagicMock,
    ) -> None:
        """Test storing documents in Chroma collection."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
            "chroma": {"collection_name": "test_collection"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_chromadb_module.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ChromaIndexingPipeline("config.yaml")

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

        # Verify collection.add was called with correct arguments
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]  # Use kwargs

        # Verify the IDs were generated properly
        assert len(call_args["ids"]) == 2
        assert all(isinstance(id_val, str) for id_val in call_args["ids"])

        # Verify embeddings were passed correctly
        assert call_args["embeddings"] == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]

        # Verify documents were passed correctly
        assert call_args["documents"] == ["Test content 1", "Test content 2"]

        # Verify metadata was passed correctly
        assert call_args["metadatas"] == [{"source": "wiki"}, {"source": "blog"}]

    @patch("vectordb.haystack.contextual_compression.indexing.chroma_indexing.chromadb")
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
        mock_chromadb_module: MagicMock,
    ) -> None:
        """Test storing documents with empty metadata."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
            "chroma": {"collection_name": "test_collection"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_chromadb_module.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ChromaIndexingPipeline("config.yaml")

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

        # Verify collection.add was called with correct metadata
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]

        # Verify metadata was handled correctly (empty dicts)
        assert call_args["metadatas"] == [{}, {}]

    @patch("vectordb.haystack.contextual_compression.indexing.chroma_indexing.chromadb")
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
        mock_chromadb_module: MagicMock,
    ) -> None:
        """Test storing documents handles failure."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
            "chroma": {"collection_name": "test_collection"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_chromadb_module.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ChromaIndexingPipeline("config.yaml")

        # Create test documents
        documents = [Document(content="Test content", meta={"source": "wiki"})]
        documents[0].embedding = [0.1, 0.2, 0.3]

        # Make collection.add raise an exception
        mock_collection.add.side_effect = Exception("Chroma storage failed")

        # Verify that the exception is propagated
        with pytest.raises(Exception, match="Chroma storage failed"):
            pipeline._store_documents(documents)

    @patch("vectordb.haystack.contextual_compression.indexing.chroma_indexing.chromadb")
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersTextEmbedder"
    )
    def test_store_documents_large_content(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_chromadb_module: MagicMock,
    ) -> None:
        """Test storing documents with large content."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
            "chroma": {"collection_name": "test_collection"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        mock_client = MagicMock()
        mock_chromadb_module.PersistentClient.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ChromaIndexingPipeline("config.yaml")

        # Create a document with very long content
        long_content = "This is a very long content. " * 1000
        documents = [Document(content=long_content, meta={"source": "wiki"})]
        documents[0].embedding = [0.1, 0.2, 0.3]

        # Call _store_documents
        pipeline._store_documents(documents)

        # Verify collection.add was called with the long content
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]

        # Verify the long content was passed correctly
        assert call_args["documents"][0] == long_content
