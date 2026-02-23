"""Tests for Chroma search pipeline."""

from unittest.mock import MagicMock, patch

import pytest


def _create_mock_config() -> MagicMock:
    """Create a properly configured mock config with logging attributes."""
    mock_config = MagicMock()
    mock_config.logging.level = "DEBUG"
    mock_config.logging.name = "test"
    return mock_config


class TestChromaSearchPipelineInit:
    """Test ChromaSearchPipeline initialization."""

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    def test_init_with_config(
        self, mock_embedder, mock_load_config, mock_client, mock_create_logger
    ) -> None:
        """Test initialization with valid config."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        # Mock config
        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        # Mock Chroma client
        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        # Mock embedder with warm_up method
        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder.return_value = mock_embedder_inst

        pipeline = ChromaSearchPipeline("test_config.yaml")

        assert pipeline.config == mock_config
        assert pipeline.collection == mock_collection
        assert pipeline.embedder == mock_embedder_inst

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    def test_init_missing_chroma_config(
        self, mock_load_config, mock_create_logger
    ) -> None:
        """Test initialization with missing Chroma config."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.chroma = None
        mock_load_config.return_value = mock_config

        with pytest.raises(ValueError, match="Chroma configuration is missing"):
            ChromaSearchPipeline("test_config.yaml")


class TestChromaSearchPipelineSearch:
    """Test ChromaSearchPipeline search functionality."""

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_basic(
        self, mock_embedder, mock_load_config, mock_client, mock_create_logger
    ) -> None:
        """Test basic search functionality."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_inst

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content1", "content2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{}, {}]],
        }

        pipeline = ChromaSearchPipeline("test_config.yaml")
        results = pipeline.search("test query", top_k=10)

        assert isinstance(results, list)
        mock_collection.query.assert_called()

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_with_empty_results(
        self, mock_embedder, mock_load_config, mock_client, mock_create_logger
    ) -> None:
        """Test search with no results."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 10
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_inst

        # Empty results
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        pipeline = ChromaSearchPipeline("test_config.yaml")
        results = pipeline.search("test query", top_k=10)

        assert isinstance(results, list)


class TestChromaSearchPipelineRAG:
    """Test RAG functionality."""

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.Pipeline")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.OpenAIGenerator"
    )
    def test_rag_generation(
        self,
        mock_gen,
        mock_pipeline,
        mock_embedder,
        mock_load_config,
        mock_client,
        mock_create_logger,
    ) -> None:
        """Test RAG generation with results."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
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

        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_inst

        # Mock Pipeline
        mock_pipeline_inst = MagicMock()
        mock_pipeline.return_value = mock_pipeline_inst

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["content1"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
        }

        pipeline = ChromaSearchPipeline("test_config.yaml")
        assert pipeline.rag_pipeline is not None


class TestChromaSearchPipelineReranking:
    """Test reranking functionality."""

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersSimilarityRanker"
    )
    def test_search_with_reranking(
        self,
        mock_ranker,
        mock_embedder,
        mock_load_config,
        mock_client,
        mock_create_logger,
    ) -> None:
        """Test search with reranking enabled."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        # Setup mocks
        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.top_k = 5
        mock_config.search.reranking_enabled = True
        mock_config.reranker.model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        mock_config.reranker.top_k = 5
        mock_config.generator.enabled = False
        mock_load_config.return_value = mock_config

        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_inst

        mock_ranker_inst = MagicMock()
        mock_ranker_inst.warm_up = MagicMock()
        mock_ranker.return_value = mock_ranker_inst

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["content1"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
        }

        pipeline = ChromaSearchPipeline("test_config.yaml")
        assert pipeline.ranker is not None


class TestChromaSearchPipelineRAGMore:
    """Extended RAG functionality tests."""

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    def test_rag_disabled_no_api_key(
        self, mock_embedder, mock_load_config, mock_client, mock_create_logger
    ) -> None:
        """Test RAG is disabled when no API key provided."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
        mock_config.embeddings.model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_config.search.reranking_enabled = False
        mock_config.generator.enabled = True
        mock_config.generator.api_key = ""  # Empty API key
        mock_load_config.return_value = mock_config

        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder.return_value = mock_embedder_inst

        pipeline = ChromaSearchPipeline("test_config.yaml")
        assert pipeline.rag_pipeline is None

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.Pipeline")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.OpenAIGenerator"
    )
    def test_search_with_rag_full(
        self,
        mock_gen,
        mock_pipeline,
        mock_embedder,
        mock_load_config,
        mock_client,
        mock_create_logger,
    ) -> None:
        """Test search_with_rag method end-to-end."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
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

        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_inst

        # Mock Pipeline
        mock_pipeline_inst = MagicMock()
        mock_pipeline_inst.run.return_value = {"llm": {"replies": ["Generated answer"]}}
        mock_pipeline.return_value = mock_pipeline_inst

        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["content1"]],
            "distances": [[0.1]],
            "metadatas": [[{"source": "test"}]],
        }

        pipeline = ChromaSearchPipeline("test_config.yaml")
        result = pipeline.search_with_rag("test query", top_k=10)

        assert "documents" in result
        assert "answer" in result
        assert result["answer"] == "Generated answer"

    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.search.chroma_searcher.Pipeline")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.OpenAIGenerator"
    )
    def test_search_with_rag_no_documents(
        self,
        mock_gen,
        mock_pipeline,
        mock_embedder,
        mock_load_config,
        mock_client,
        mock_create_logger,
    ) -> None:
        """Test search_with_rag when no documents found."""
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearchPipeline,
        )

        mock_config = _create_mock_config()
        mock_config.chroma.path = "/tmp/chroma"
        mock_config.collection.name = "test_collection"
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

        mock_client_inst = MagicMock()
        mock_collection = MagicMock()
        mock_client_inst.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_inst

        mock_embedder_inst = MagicMock()
        mock_embedder_inst.warm_up = MagicMock()
        mock_embedder_inst.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_inst

        # Mock Pipeline
        mock_pipeline_inst = MagicMock()
        mock_pipeline.return_value = mock_pipeline_inst

        # Mock empty search results
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        pipeline = ChromaSearchPipeline("test_config.yaml")
        result = pipeline.search_with_rag("test query", top_k=10)

        assert "documents" in result
        assert "answer" in result
        assert result["answer"] is None
        assert len(result["documents"]) == 0
