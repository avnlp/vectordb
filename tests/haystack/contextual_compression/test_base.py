"""Comprehensive tests for BaseContextualCompressionPipeline.

Tests cover:
- Pipeline initialization
- Embedder setup with different configurations
- Compressor initialization
- Run method behavior
- Evaluate method
- Error handling
"""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.base import (
    BaseContextualCompressionPipeline,
)


class ConcreteCompressionPipeline(BaseContextualCompressionPipeline):
    """Concrete implementation of BaseContextualCompressionPipeline for testing."""

    def _connect(self) -> None:
        """Mock connection implementation."""
        self.connection_mock = MagicMock()

    def _ensure_collection_ready(self) -> None:
        """Mock collection preparation."""
        self.collection_mock = MagicMock()

    def _retrieve_base_results(self, query: str, top_k: int) -> list[Document]:
        """Mock document retrieval."""
        return [
            Document(content=f"Result for: {query}", meta={"id": "1"}),
            Document(content="Another result", meta={"id": "2"}),
        ]


class TestBaseContextualCompressionPipelineInit:
    """Tests for pipeline initialization."""

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch.object(ConcreteCompressionPipeline, "_init_compressor")
    def test_initialization(
        self,
        mock_init_compressor: MagicMock,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test pipeline initialization sequence."""
        mock_config = {
            "embeddings": {"model": "all-MiniLM-L6-v2"},
            "compression": {"type": "reranking"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        pipeline = ConcreteCompressionPipeline("config.yaml")

        # Verify initialization sequence
        mock_load_config.assert_called_once_with("config.yaml")
        mock_setup_logger.assert_called_once_with(mock_config)
        assert pipeline.config == mock_config
        assert pipeline.logger == mock_logger

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_init_embedders_default_model(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test embedder initialization with default model."""
        mock_config = {"embeddings": {}}
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        with patch.object(ConcreteCompressionPipeline, "_init_compressor"):
            pipeline = ConcreteCompressionPipeline("config.yaml")

        # Default model should be Qwen/Qwen3-Embedding-0.6B
        mock_embedder_class.assert_called_once_with(model="Qwen/Qwen3-Embedding-0.6B")
        mock_embedder_instance.warm_up.assert_called_once()
        assert pipeline.dense_embedder == mock_embedder_instance
        mock_logger.info.assert_called_with(
            "Initialized dense embedder with model: %s", "Qwen/Qwen3-Embedding-0.6B"
        )

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_init_embedders_qwen3_alias(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test embedder initialization with qwen3 alias."""
        mock_config = {"embeddings": {"model": "qwen3"}}
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        with patch.object(ConcreteCompressionPipeline, "_init_compressor"):
            ConcreteCompressionPipeline("config.yaml")

        mock_embedder_class.assert_called_once_with(model="Qwen/Qwen3-Embedding-0.6B")

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_init_embedders_minilm_alias(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test embedder initialization with minilm alias."""
        mock_config = {"embeddings": {"model": "minilm"}}
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        with patch.object(ConcreteCompressionPipeline, "_init_compressor"):
            ConcreteCompressionPipeline("config.yaml")

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_init_embedders_custom_model(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test embedder initialization with custom model."""
        mock_config = {"embeddings": {"model": "custom-model-name"}}
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        with patch.object(ConcreteCompressionPipeline, "_init_compressor"):
            ConcreteCompressionPipeline("config.yaml")

        mock_embedder_class.assert_called_once_with(model="custom-model-name")


class TestBaseContextualCompressionPipelineCompressor:
    """Tests for compressor initialization."""

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    def test_init_compressor_success(
        self,
        mock_factory: MagicMock,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test successful compressor initialization."""
        mock_config = {
            "embeddings": {},
            "compression": {"type": "reranking"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_compressor = MagicMock()
        mock_factory.create_compressor.return_value = mock_compressor

        pipeline = ConcreteCompressionPipeline("config.yaml")

        mock_factory.create_compressor.assert_called_once_with(mock_config)
        assert pipeline.compressor == mock_compressor
        mock_logger.info.assert_any_call("Initialized compressor: %s", "reranking")

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    def test_init_compressor_failure(
        self,
        mock_factory: MagicMock,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test compressor initialization failure handling."""
        mock_config = {
            "embeddings": {},
            "compression": {"type": "reranking"},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_factory.create_compressor.side_effect = ValueError("Invalid config")

        with pytest.raises(ValueError, match="Invalid config"):
            ConcreteCompressionPipeline("config.yaml")

        mock_logger.error.assert_called_with(
            "Failed to initialize compressor: %s", "Invalid config"
        )


class TestBaseContextualCompressionPipelineRun:
    """Tests for the run method."""

    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_run_successful(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_factory: MagicMock,
    ) -> None:
        """Test successful run with compression."""
        mock_config = {
            "embeddings": {},
            "retrieval": {"top_k": 20},
            "compression": {"type": "reranking", "reranker": {"type": "cross_encoder"}},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock compressor
        mock_compressor = MagicMock()
        mock_compressor.run.return_value = {
            "documents": [
                Document(content="Compressed 1", meta={"id": "1"}),
                Document(content="Compressed 2", meta={"id": "2"}),
            ]
        }
        mock_factory.create_compressor.return_value = mock_compressor

        pipeline = ConcreteCompressionPipeline("config.yaml")

        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        assert len(result["documents"]) == 2
        mock_logger.info.assert_any_call(
            "Running compression pipeline for query: %s", "test query"
        )
        mock_logger.info.assert_any_call("Compressed %d documents to %d", 2, 2)

    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_run_no_documents(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_factory: MagicMock,
    ) -> None:
        """Test run when no documents are retrieved."""
        mock_config = {
            "embeddings": {},
            "retrieval": {"top_k": 10},
            "compression": {"type": "reranking", "reranker": {"type": "cross_encoder"}},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_factory.create_compressor.return_value = MagicMock()

        pipeline = ConcreteCompressionPipeline("config.yaml")

        # Override _retrieve_base_results to return empty
        pipeline._retrieve_base_results = MagicMock(return_value=[])

        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}
        mock_logger.info.assert_any_call("No documents retrieved")

    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_run_with_error(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_factory: MagicMock,
    ) -> None:
        """Test run when compressor raises an error."""
        mock_config = {
            "embeddings": {},
            "retrieval": {"top_k": 10},
            "compression": {"type": "reranking", "reranker": {"type": "cross_encoder"}},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance

        # Mock compressor to raise error
        mock_compressor = MagicMock()
        mock_compressor.run.side_effect = Exception("Compression failed")
        mock_factory.create_compressor.return_value = mock_compressor

        pipeline = ConcreteCompressionPipeline("config.yaml")

        result = pipeline.run("test query", top_k=5)

        assert result == {"documents": []}
        mock_logger.error.assert_called_with(
            "Error during compression pipeline: %s", "Compression failed"
        )

    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_run_default_top_k(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_factory: MagicMock,
    ) -> None:
        """Test run with default top_k value."""
        mock_config = {
            "embeddings": {},
            "retrieval": {},
            "compression": {"type": "reranking", "reranker": {"type": "cross_encoder"}},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_factory.create_compressor.return_value = MagicMock()

        pipeline = ConcreteCompressionPipeline("config.yaml")

        # Mock _retrieve_base_results to capture the top_k value
        pipeline._retrieve_base_results = MagicMock(return_value=[])

        # Call without specifying top_k (should default to 10)
        pipeline.run("test query")

        # retrieval_top_k should be top_k * 2 = 20
        pipeline._retrieve_base_results.assert_called_once_with("test query", 20)


class TestBaseContextualCompressionPipelineEvaluate:
    """Tests for the evaluate method."""

    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_evaluate_basic(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_factory: MagicMock,
    ) -> None:
        """Test basic evaluation functionality."""
        mock_config = {
            "embeddings": {},
            "compression": {"type": "reranking", "reranker": {"type": "cross_encoder"}},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_factory.create_compressor.return_value = MagicMock()

        pipeline = ConcreteCompressionPipeline("config.yaml")
        pipeline.run = MagicMock(return_value={"documents": [MagicMock()]})

        questions = ["What is AI?", "How does ML work?"]
        ground_truths = ["AI is...", "ML works by..."]

        result = pipeline.evaluate(questions, ground_truths)

        assert result["questions"] == 2
        assert "metrics" in result
        assert pipeline.run.call_count == 2
        mock_logger.info.assert_any_call("Evaluating pipeline on %d questions", 2)
        mock_logger.info.assert_any_call("Evaluation completed for %d questions", 2)

    @patch("vectordb.haystack.contextual_compression.base.CompressorFactory")
    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_evaluate_empty_questions(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
        mock_factory: MagicMock,
    ) -> None:
        """Test evaluation with empty questions."""
        mock_config = {
            "embeddings": {},
            "compression": {"type": "reranking", "reranker": {"type": "cross_encoder"}},
        }
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger
        mock_embedder_instance = MagicMock()
        mock_embedder_class.return_value = mock_embedder_instance
        mock_factory.create_compressor.return_value = MagicMock()

        pipeline = ConcreteCompressionPipeline("config.yaml")
        pipeline.run = MagicMock()

        result = pipeline.evaluate([], [])

        assert result["questions"] == 0
        assert pipeline.run.call_count == 0


class TestBaseContextualCompressionPipelineAbstract:
    """Tests to verify abstract methods are required."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test BaseContextualCompressionPipeline instantiation.

        Cannot be instantiated directly.
        """
        with pytest.raises(TypeError):
            BaseContextualCompressionPipeline("config.yaml")

    @patch("vectordb.haystack.contextual_compression.base.load_config")
    @patch("vectordb.haystack.contextual_compression.base.setup_logger")
    @patch(
        "vectordb.haystack.contextual_compression.base.SentenceTransformersTextEmbedder"
    )
    def test_missing_abstract_methods(
        self,
        mock_embedder_class: MagicMock,
        mock_setup_logger: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test that missing abstract methods raise TypeError."""
        mock_config = {"embeddings": {}}
        mock_load_config.return_value = mock_config
        mock_setup_logger.return_value = MagicMock()
        mock_embedder_class.return_value = MagicMock()

        class IncompletePipeline(BaseContextualCompressionPipeline):
            missing = True

        with pytest.raises(TypeError):
            IncompletePipeline("config.yaml")
