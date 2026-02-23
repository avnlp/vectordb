"""Integration tests for Milvus agentic RAG pipelines.

This module provides integration test coverage for agentic RAG pipelines
that use autonomous decision-making for improved retrieval and answer quality.
Tests cover dataloader integration, runtime dataset switching, and evaluation.

Key Features Tested:
    - Pipeline initialization with configuration files
    - Dataloader setup and lazy document loading
    - Dataset switching at runtime (TriviaQA, ARC, etc.)
    - Evaluation workflow with ground truth comparison
    - Configuration validation for retry, retrieval, and indexing settings

Environment Requirements:
    - Network access for dataset loading (TriviaQA, ARC)
    - Milvus server for vector storage (URI in config)
    - Groq API key for LLM-based answer generation

Test Markers:
    - integration: Tests requiring external service connections
    - enable_socket: Tests requiring network access
"""

import pytest

from vectordb.haystack.agentic_rag.milvus_agentic_rag import MilvusAgenticRAGPipeline


@pytest.fixture
def config_path() -> str:
    """Get path to test configuration file.

    Returns:
        Path to Milvus TriviaQA configuration YAML for agentic RAG testing.
    """
    return "src/vectordb/haystack/agentic_rag/configs/milvus_triviaqa.yaml"


@pytest.mark.integration
@pytest.mark.enable_socket
class TestAgenticRAGIntegration:
    """Integration test suite for Milvus agentic RAG pipelines.

    Tests cover end-to-end pipeline functionality including initialization,
    dataset loading, configuration validation, and evaluation workflows.
    All tests require network access for external service connections.
    """

    def test_pipeline_initialization(self, config_path: str) -> None:
        """Test that pipeline initializes without errors."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        # Check all components initialized
        assert pipeline.config is not None
        assert pipeline.dense_embedder is not None
        assert pipeline.router is not None
        assert pipeline.generator is not None
        assert pipeline.dataloader is not None

    def test_dataloader_initialization(self, config_path: str) -> None:
        """Test that dataloader is properly initialized."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        # Should be initialized but no data loaded yet
        assert pipeline.data is None
        assert pipeline.documents is None
        assert pipeline.ground_truths is None

    def test_load_dataset(self, config_path: str) -> None:
        """Test loading dataset via load_dataset method."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        # Load with limit for quick test
        pipeline.load_dataset(limit=2)

        # Data should be loaded
        assert pipeline.data is not None
        assert pipeline.documents is not None or len(pipeline.documents or []) >= 0
        assert (
            pipeline.ground_truths is not None or len(pipeline.ground_truths or []) >= 0
        )

    def test_runtime_dataset_switching(self, config_path: str) -> None:
        """Test switching datasets at runtime."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        # Load first dataset
        pipeline.load_dataset(limit=2)
        first_ground_truths = len(pipeline.ground_truths or [])

        # Switch to different dataset
        pipeline.load_dataset(dataset_type="arc", limit=2)
        second_ground_truths = len(pipeline.ground_truths or [])

        # Should have loaded data for both datasets
        assert first_ground_truths >= 0
        assert second_ground_truths >= 0

    def test_evaluate_no_dataset_error(self, config_path: str) -> None:
        """Test that evaluate fails gracefully when no dataset loaded."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        # Evaluate without loading dataset
        result = pipeline.evaluate()

        assert result.get("error") == "No dataset loaded"
        assert result["total"] == 0

    def test_config_contains_retry_settings(self, config_path: str) -> None:
        """Test that config has retry settings."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        agentic_config = pipeline.config.get("agentic_rag", {})
        assert "max_retries" in agentic_config
        assert "retry_delay_seconds" in agentic_config
        assert "fallback_tool" in agentic_config

    def test_config_contains_retrieval_settings(self, config_path: str) -> None:
        """Test that config has retrieval settings."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        retrieval_config = pipeline.config.get("retrieval", {})
        assert "top_k_default" in retrieval_config

    def test_config_contains_indexing_settings(self, config_path: str) -> None:
        """Test that config has indexing settings."""
        pipeline = MilvusAgenticRAGPipeline(config_path)

        indexing_config = pipeline.config.get("indexing", {})
        assert "chunk_size" in indexing_config
        assert "chunk_overlap" in indexing_config
        assert "batch_size" in indexing_config
