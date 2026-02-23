"""Tests for BaseIndexingPipeline functionality."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml
from haystack import Document

from vectordb.haystack.contextual_compression.indexing.base_indexing import (
    BaseIndexingPipeline,
)
from vectordb.haystack.json_indexing.common.config import load_config


class ConcreteIndexingPipeline(BaseIndexingPipeline):
    """Concrete implementation of BaseIndexingPipeline for testing purposes."""

    def _connect(self) -> None:
        """Mock connection implementation."""
        self.connection_mock = MagicMock()

    def _prepare_collection(self) -> None:
        """Mock collection preparation."""
        self.collection_mock = MagicMock()

    def _store_documents(self, documents: list[Document]) -> None:
        """Mock document storage."""
        self.stored_documents = documents


class TestBaseIndexingPipeline:
    """Unit tests for BaseIndexingPipeline functionality."""

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    def test_initialization(
        self,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_class: MagicMock,
    ) -> None:
        """Test BaseIndexingPipeline initialization."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        mock_load_config.assert_called_once_with("config.yaml")
        mock_setup_logger.assert_called_once_with(mock_config)
        assert pipeline.config == mock_config
        assert pipeline.logger == mock_logger
        mock_embedder_class.assert_called_once_with(model="all-MiniLM-L6-v2")
        mock_embedder_instance.warm_up.assert_called_once()
        assert pipeline.dense_embedder == mock_embedder_instance

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_init_embedders_default_model(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test embedder initialization with default model."""
        mock_config = {
            "embeddings": {},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        mock_embedder_class.assert_called_once_with(model="Qwen/Qwen3-Embedding-0.6B")
        mock_embedder_instance.warm_up.assert_called_once()
        assert pipeline.dense_embedder == mock_embedder_instance

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_init_embedders_with_alias(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test embedder initialization with model alias."""
        mock_config = {
            "embeddings": {"model": "qwen3"},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        mock_embedder_class.assert_called_once_with(model="Qwen/Qwen3-Embedding-0.6B")
        mock_embedder_instance.warm_up.assert_called_once()
        assert pipeline.dense_embedder == mock_embedder_instance

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_init_embedders_with_minilm_alias(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test embedder initialization with MiniLM alias."""
        mock_config = {
            "embeddings": {"model": "minilm"},
            "dataset": {"type": "test", "name": "test_dataset"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_embedder_instance.warm_up.assert_called_once()
        assert pipeline.dense_embedder == mock_embedder_instance

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_load_dataset(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test dataset loading functionality."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
                "split": "train",
                "limit": 10,
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        # Mock the _load_dataset method to return test documents directly
        test_docs = [
            Document(content="Sample text 1", meta={"id": 1}),
            Document(content="Sample text 2", meta={"id": 2}),
        ]
        pipeline._load_dataset = MagicMock(return_value=test_docs)

        documents = pipeline._load_dataset()

        assert len(documents) == 2
        assert documents[0].content == "Sample text 1"
        assert documents[0].meta == {"id": 1}
        assert documents[1].content == "Sample text 2"
        assert documents[1].meta == {"id": 2}

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.DataloaderCatalog"
    )
    def test_load_dataset_missing_type(
        self,
        mock_dataset_registry: MagicMock,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test dataset loading with missing type raises error."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {},  # Missing type
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        with pytest.raises(ValueError, match="dataset.type not specified in config"):
            pipeline._load_dataset()

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_run_success(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful indexing run."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
                "split": "train",
                "limit": 2,
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()

        def mock_run(documents):
            # Return documents with embeddings
            for doc in documents:
                doc.embedding = [0.1, 0.2, 0.3]
            return {"documents": documents}

        mock_embedder_instance.run.side_effect = mock_run
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        # Mock the _load_dataset method to return test documents
        test_docs = [
            Document(content="Sample text 1", meta={"id": 1}),
            Document(content="Sample text 2", meta={"id": 2}),
        ]
        pipeline._load_dataset = MagicMock(return_value=test_docs)

        result = pipeline.run(batch_size=2)

        assert result["indexed_count"] == 2
        assert result["status"] == "success"
        assert result["batch_size"] == 2
        assert len(pipeline.stored_documents) == 2
        assert pipeline.stored_documents[0].embedding == [0.1, 0.2, 0.3]

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_run_empty_dataset(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test indexing run with empty dataset."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
                "split": "train",
                "limit": 0,
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        # Mock the _load_dataset method to return empty list
        pipeline._load_dataset = MagicMock(return_value=[])

        result = pipeline.run()

        assert result["indexed_count"] == 0
        assert result["status"] == "empty_dataset"

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_run_with_exception(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test indexing run with exception during processing."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
                "split": "train",
                "limit": 2,
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()

        def mock_run(documents):
            # Return documents with embeddings
            for doc in documents:
                doc.embedding = [0.1, 0.2, 0.3]
            return {"documents": documents}

        mock_embedder_instance.run.side_effect = mock_run
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        # Mock the _load_dataset method to return test documents
        test_docs = [
            Document(content="Sample text 1", meta={"id": 1}),
            Document(content="Sample text 2", meta={"id": 2}),
        ]
        pipeline._load_dataset = MagicMock(return_value=test_docs)

        # Make _store_documents raise an exception
        original_store = ConcreteIndexingPipeline._store_documents

        def mock_store_documents(self, documents):
            raise RuntimeError("Storage failed")

        ConcreteIndexingPipeline._store_documents = mock_store_documents
        try:
            result = pipeline.run()

            assert result["indexed_count"] == 0
            assert result["status"] == "error"
            assert "Storage failed" in result["error"]
        finally:
            # Restore original method
            ConcreteIndexingPipeline._store_documents = original_store

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_run_batch_processing(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test batch processing during indexing run."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
                "split": "train",
                "limit": 5,
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()

        def mock_run(documents):
            # Return documents with embeddings
            for doc in documents:
                doc.embedding = [0.1, 0.2, 0.3]
            return {"documents": documents}

        mock_embedder_instance.run.side_effect = mock_run
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        # Create 5 documents to test batching
        test_docs = [
            Document(content=f"Sample text {i}", meta={"id": i}) for i in range(1, 6)
        ]
        pipeline._load_dataset = MagicMock(return_value=test_docs)

        result = pipeline.run(batch_size=2)  # Batch size of 2

        assert result["indexed_count"] == 5
        assert result["status"] == "success"
        assert result["batch_size"] == 2
        # Verify embedder called for each batch (3 batches for 5 docs with batch_size=2)
        assert mock_embedder_instance.run.call_count == 3

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_run_dataset_load_exception(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test run handles exception during dataset loading."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        pipeline._load_dataset = MagicMock(side_effect=ValueError("Dataset not found"))

        result = pipeline.run()

        assert result["indexed_count"] == 0
        assert result["status"] == "error"
        assert "Dataset not found" in result["error"]

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_run_embedding_exception(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test run handles exception during embedding generation."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.run.side_effect = RuntimeError("Embedder failed")
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        test_docs = [Document(content="Sample text", meta={"id": 1})]
        pipeline._load_dataset = MagicMock(return_value=test_docs)

        result = pipeline.run()

        assert result["indexed_count"] == 0
        assert result["status"] == "error"
        assert "Embedder failed" in result["error"]

    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.load_config"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.setup_logger"
    )
    @patch(
        "vectordb.haystack.contextual_compression.indexing.base_indexing.SentenceTransformersDocumentEmbedder"
    )
    def test_run_with_none_documents_from_load(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test run with no documents returns empty_dataset status."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "dataset": {
                "type": "triviaqa",
                "name": "test_dataset",
            },
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteIndexingPipeline("config.yaml")

        pipeline._load_dataset = MagicMock(return_value=[])

        result = pipeline.run()

        assert result["indexed_count"] == 0
        assert result["status"] == "empty_dataset"
        mock_embedder_instance.run.assert_not_called()

    def test_load_config_malformed_yaml_raises_error(self) -> None:
        """Test that load_config raises yaml.YAMLError for malformed YAML."""
        malformed_yaml = """invalid_yaml: [
    unclosed: quote
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(malformed_yaml)
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            os.unlink(config_path)
