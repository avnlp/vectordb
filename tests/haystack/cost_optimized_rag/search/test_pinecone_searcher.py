"""Tests for Pinecone search pipeline."""

from unittest.mock import MagicMock, patch

import pytest


def _create_mock_config() -> MagicMock:
    """Create a properly configured mock config with logging attributes."""
    mock_config = MagicMock()
    mock_config.logging.level = "DEBUG"
    mock_config.logging.name = "test"
    return mock_config


class TestPineconeSearchPipelineInit:
    """Test PineconeSearchPipeline initialization."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_init_with_config(
        self, mock_embedder, mock_load_config, mock_pinecone, mock_create_logger
    ) -> None:
        """Test initialization with valid config."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        # Mock config
        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        # Mock Pinecone client
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        # Mock embedder with warm_up method
        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder.return_value = mock_embedder_inst

        pipeline = PineconeSearchPipeline("test_config.yaml")

        assert pipeline.config == mock_config
        assert pipeline.index == mock_index
        assert pipeline.embedder == mock_embedder_inst

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    def test_init_missing_pinecone_config(
        self, mock_load_config, mock_create_logger
    ) -> None:
        """Test initialization with missing Pinecone config."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.pinecone = None
        mock_load_config.return_value = mock_config

        with pytest.raises(ValueError, match="Pinecone configuration is missing"):
            PineconeSearchPipeline("test_config.yaml")


class TestPineconeSearchPipelineSearch:
    """Test PineconeSearchPipeline search functionality."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_basic(
        self, mock_embedder, mock_load_config, mock_pinecone, mock_create_logger
    ) -> None:
        """Test basic search functionality."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 1536}
        mock_embedder.return_value = mock_embedder_inst

        # Mock search results
        mock_index.query.return_value = {
            "matches": [
                {"id": "doc1", "score": 0.9, "metadata": {"content": "content1"}},
                {"id": "doc2", "score": 0.8, "metadata": {"content": "content2"}},
            ]
        }

        pipeline = PineconeSearchPipeline("test_config.yaml")
        results = pipeline.search("test query", top_k=10)

        assert isinstance(results, list)
        mock_index.query.assert_called()

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_with_empty_results(
        self, mock_embedder, mock_load_config, mock_pinecone, mock_create_logger
    ) -> None:
        """Test search with no results."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 1536}
        mock_embedder.return_value = mock_embedder_inst

        # Empty results
        mock_index.query.return_value = {"matches": []}

        pipeline = PineconeSearchPipeline("test_config.yaml")
        results = pipeline.search("test query", top_k=10)

        assert isinstance(results, list)

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_with_namespaces(
        self, mock_embedder, mock_load_config, mock_pinecone, mock_create_logger
    ) -> None:
        """Test search with namespace parameter."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 1536}
        mock_embedder.return_value = mock_embedder_inst

        # Mock search results
        mock_index.query.return_value = {"matches": []}

        pipeline = PineconeSearchPipeline("test_config.yaml")
        results = pipeline.search("test query", top_k=10)

        assert isinstance(results, list)


class TestPineconeSearchPipelineReranking:
    """Test reranking functionality."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersSimilarityRanker"
    )
    def test_search_with_reranking(
        self,
        mock_ranker,
        mock_embedder,
        mock_load_config,
        mock_pinecone,
        mock_create_logger,
    ) -> None:
        """Test search with reranking enabled."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = True
        mock_config.reranker.model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        mock_config.reranker.top_k = 5
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 1536}
        mock_embedder.return_value = mock_embedder_inst

        mock_ranker_inst = MagicMock()
        mock_ranker_inst.warm_up = MagicMock()
        mock_ranker.return_value = mock_ranker_inst

        # Mock search results
        mock_index.query.return_value = {
            "matches": [
                {"id": "doc1", "score": 0.9, "metadata": {"content": "content1"}}
            ]
        }

        pipeline = PineconeSearchPipeline("test_config.yaml")
        assert pipeline.ranker is not None


class TestPineconeSearchPipelineRAG:
    """Extended RAG functionality tests."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_rag_disabled_no_api_key(
        self, mock_embedder, mock_load_config, mock_pinecone, mock_create_logger
    ) -> None:
        """Test RAG is disabled when no API key provided."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = True
        mock_config.generator.api_key = ""  # Empty API key
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder.return_value = mock_embedder_inst

        pipeline = PineconeSearchPipeline("test_config.yaml")
        assert pipeline.rag_pipeline is None

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pipeline")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.OpenAIGenerator"
    )
    def test_search_with_rag_full(
        self,
        mock_gen,
        mock_pipeline,
        mock_embedder,
        mock_load_config,
        mock_pinecone,
        mock_create_logger,
    ) -> None:
        """Test search_with_rag method end-to-end."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = True
        mock_config.generator.api_key = "test-key"
        mock_config.generator.api_base_url = "https://api.openai.com/v1"
        mock_config.generator.model = "gpt-3.5-turbo"
        mock_config.generator.temperature = 0.7
        mock_config.generator.max_tokens = 256
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 1536}
        mock_embedder.return_value = mock_embedder_inst

        # Mock Pipeline
        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.run.return_value = {"llm": {"replies": ["Generated answer"]}}
        mock_pipeline.return_value = mock_pipeline_inst

        # Mock search results
        mock_index.query.return_value = {
            "matches": [
                {"id": "doc1", "score": 0.9, "metadata": {"content": "content1"}}
            ]
        }

        pipeline = PineconeSearchPipeline("test_config.yaml")
        result = pipeline.search_with_rag("test query", top_k=10)

        assert "documents" in result
        assert "answer" in result
        assert result["answer"] == "Generated answer"

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pipeline")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.OpenAIGenerator"
    )
    def test_search_with_rag_no_documents(
        self,
        mock_gen,
        mock_pipeline,
        mock_embedder,
        mock_load_config,
        mock_pinecone,
        mock_create_logger,
    ) -> None:
        """Test search_with_rag when no documents found."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = True
        mock_config.generator.api_key = "test-key"
        mock_config.generator.api_base_url = "https://api.openai.com/v1"
        mock_config.generator.model = "gpt-3.5-turbo"
        mock_config.generator.temperature = 0.7
        mock_config.generator.max_tokens = 256
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 1536}
        mock_embedder.return_value = mock_embedder_inst

        # Mock Pipeline
        mock_pipeline_inst = MagicMock()
        mock_pipeline.return_value = mock_pipeline_inst

        # Mock empty search results
        mock_index.query.return_value = {"matches": []}

        pipeline = PineconeSearchPipeline("test_config.yaml")
        result = pipeline.search_with_rag("test query", top_k=10)

        assert "documents" in result
        assert "answer" in result
        assert result["answer"] is None
        assert len(result["documents"]) == 0

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_metadata_extraction(
        self, mock_embedder, mock_load_config, mock_pinecone, mock_create_logger
    ) -> None:
        """Test that metadata is properly extracted from search results."""
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.pinecone.api_key = "test-key"
        mock_config.collection.name = "test-index"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 1536}
        mock_embedder.return_value = mock_embedder_inst

        # Mock search results with metadata
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.9,
                    "metadata": {
                        "content": "test content",
                        "source": "test_source",
                        "author": "test_author",
                    },
                }
            ]
        }

        pipeline = PineconeSearchPipeline("test_config.yaml")
        results = pipeline.search("test query", top_k=10)

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "test content"
        assert results[0]["metadata"]["source"] == "test_source"
        assert results[0]["metadata"]["author"] == "test_author"
