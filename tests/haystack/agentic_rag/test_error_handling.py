"""Tests for error handling and edge cases in agentic RAG."""

from unittest.mock import Mock, patch

import pytest
from haystack import Document

from vectordb.haystack.agentic_rag.base import BaseAgenticRAGPipeline


class MockAgenticRAGPipeline(BaseAgenticRAGPipeline):
    """Mock implementation of BaseAgenticRAGPipeline for testing purposes."""

    def __init__(self, config_path: str = "/fake/path"):
        """Don't call parent constructor to avoid API calls during initialization."""
        # We'll set attributes manually in the fixture
        self.client = None

    def _connect(self) -> None:
        """Mock connection method."""
        self.client = Mock()

    def _create_index(self) -> None:
        """Mock index creation method."""
        self.collection = Mock()

    def index_documents(self) -> int:
        """Mock index documents method."""
        return 0

    def _retrieve(self, query: str, top_k: int) -> list[Document]:
        """Mock retrieve method."""
        return []


class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""

    @pytest.fixture
    def mock_pipeline(self, mock_config):
        """Create a mock pipeline instance for testing."""
        pipeline = MockAgenticRAGPipeline("/fake/path")
        pipeline.config = mock_config
        pipeline.logger = Mock()
        pipeline.router = Mock()
        pipeline.generator = Mock()
        pipeline.dataloader = Mock()
        pipeline.data = None
        pipeline.documents = None
        pipeline.ground_truths = None
        return pipeline

    def test_router_initialization_failure(self, mock_pipeline):
        """Test router initialization failure handling."""
        with (
            patch(
                "vectordb.haystack.agentic_rag.base.AgenticRouter",
                side_effect=Exception("router error"),
            ),
            pytest.raises(Exception, match="router error"),
        ):
            mock_pipeline._init_router()

        mock_pipeline.logger.error.assert_called_once_with(
            "Failed to initialize AgenticRouter: %s",
            "router error",
        )

    def test_generator_initialization_uses_env_key(self, mock_pipeline):
        """Test generator initialization uses environment API key fallback."""
        mock_pipeline.config["generator"].pop("api_key", None)
        generator_instance = Mock()

        with (
            patch("os.getenv", return_value="env-key"),
            patch(
                "vectordb.haystack.agentic_rag.base.Secret.from_token",
                return_value="secret-token",
            ),
            patch(
                "vectordb.haystack.agentic_rag.base.OpenAIGenerator",
                return_value=generator_instance,
            ) as mock_generator,
        ):
            mock_pipeline._init_generator()

        mock_generator.assert_called_once()
        called_kwargs = mock_generator.call_args.kwargs
        assert called_kwargs["api_key"] == "secret-token"
        assert called_kwargs["model"] == "test-generator"
        generator_instance.warm_up.assert_called_once()

    def test_generator_initialization_failure(self, mock_pipeline):
        """Test generator initialization failure handling."""
        with (
            patch(
                "vectordb.haystack.agentic_rag.base.OpenAIGenerator",
                side_effect=Exception("generator error"),
            ),
            pytest.raises(Exception, match="generator error"),
        ):
            mock_pipeline._init_generator()

        mock_pipeline.logger.error.assert_called_once_with(
            "Failed to initialize generator: %s",
            "generator error",
        )

    def test_dataloader_initialization_failure(self, mock_pipeline):
        """Test dataloader initialization failure handling."""
        with patch(
            "vectordb.haystack.agentic_rag.base.get_dataloader_instance",
            side_effect=Exception("load error"),
        ):
            mock_pipeline._load_dataloader()

        assert mock_pipeline.dataloader is None
        assert mock_pipeline.documents is None
        mock_pipeline.logger.warning.assert_called_once_with(
            "Failed to initialize dataloader: %s",
            "load error",
        )

    def test_load_dataset_applies_limit(self, mock_pipeline):
        """Test load_dataset applies limit to documents."""
        mock_loader = Mock()
        mock_loader.load_data.return_value = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        mock_loader.get_documents.return_value = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
            Document(content="Doc 3"),
        ]

        with patch(
            "vectordb.haystack.agentic_rag.base.get_dataloader_instance",
            return_value=mock_loader,
        ):
            mock_pipeline.load_dataset(limit=2)

        assert len(mock_pipeline.documents) == 2
        assert len(mock_pipeline.ground_truths) == 2

    def test_load_dataset_dataloader_error(self, mock_pipeline):
        """Test load_dataset raises when dataloader fails."""
        with (
            patch(
                "vectordb.haystack.agentic_rag.base.get_dataloader_instance",
                side_effect=Exception("bad dataloader"),
            ),
            pytest.raises(Exception, match="bad dataloader"),
        ):
            mock_pipeline.load_dataset()

        mock_pipeline.logger.error.assert_called_once_with(
            "Failed to load dataloader: %s",
            "bad dataloader",
        )

    def test_extract_ground_truths_skips_non_dict(self, mock_pipeline):
        """Test extracting ground truths skips non-dict items."""
        mock_pipeline.data = [
            "not-a-dict",
            {"question": "Valid question", "answer": "Valid answer"},
        ]

        ground_truths = mock_pipeline._extract_ground_truths()

        assert ground_truths == [
            {"question": "Valid question", "answer": "Valid answer"}
        ]

    @pytest.mark.parametrize(
        ("tool", "handler_name"),
        [
            ("web_search", "_handle_web_search"),
            ("calculation", "_handle_calculation"),
            ("reasoning", "_handle_reasoning"),
        ],
    )
    def test_run_routes_tool_handlers(self, mock_pipeline, tool, handler_name):
        """Test run routes to the expected tool handler."""
        mock_pipeline._get_routing_enabled = Mock(return_value=True)
        mock_pipeline.router.select_tool = Mock(return_value=tool)
        handler = Mock(return_value={"documents": [], "answer": "ok", "tool": tool})
        setattr(mock_pipeline, handler_name, handler)

        result = mock_pipeline.run("query", top_k=3)

        handler.assert_called_once()
        assert result["tool"] == tool

    def test_run_with_self_reflection_enabled(self, mock_pipeline):
        """Test run executes self-reflection loop when enabled."""
        mock_pipeline._get_routing_enabled = Mock(return_value=False)
        mock_pipeline._get_self_reflection_enabled = Mock(return_value=True)
        mock_pipeline._get_max_iterations = Mock(return_value=2)
        mock_pipeline._get_quality_threshold = Mock(return_value=80)
        mock_pipeline._handle_retrieval = Mock(
            return_value={
                "documents": [Document(content="Doc 1")],
                "answer": "Initial answer",
                "tool": "retrieval",
            }
        )
        mock_pipeline.router.self_reflect_loop = Mock(return_value="Refined answer")

        result = mock_pipeline.run("query", top_k=2)

        assert result["answer"] == "Refined answer"
        assert result["refined"] is True
        mock_pipeline.router.self_reflect_loop.assert_called_once()

    def test_evaluate_calculates_averages(self, mock_pipeline):
        """Test evaluate calculates average documents and refinement rate."""
        mock_pipeline.run = Mock(
            side_effect=[
                {
                    "answer": "A1",
                    "tool": "retrieval",
                    "documents": [Document(content="D1"), Document(content="D2")],
                    "refined": True,
                },
                {
                    "answer": "A2",
                    "tool": "calculation",
                    "documents": [Document(content="D3")],
                    "refined": False,
                },
            ]
        )

        result = mock_pipeline.evaluate(questions=["Q1", "Q2"])

        assert result["metrics"]["documents_retrieved"] == 3
        assert result["metrics"]["avg_documents"] == 1.5
        assert result["metrics"]["refinement_rate"] == 0.5
        assert result["metrics"]["tools_used"]["retrieval"] == 1
        assert result["metrics"]["tools_used"]["calculation"] == 1


# Fixtures for individual tests
@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
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
    }


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
def mock_qdrant_config():
    """Mock configuration for Qdrant testing."""
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
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "api_key": "test-api-key",
        },
        "collection": {"name": "test-collection"},
    }


@pytest.fixture
def mock_weaviate_config():
    """Mock configuration for Weaviate testing."""
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
        "weaviate": {
            "host": "localhost",
            "port": 8080,
            "grpc_port": 50051,
            "api_key": "test-api-key",
        },
        "collection": {"name": "TestCollection"},
    }


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
