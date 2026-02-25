"""Tests for Qdrant agentic RAG search pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest


class TestQdrantAgenticRAGSearch:
    """Tests for QdrantAgenticRAGPipeline search functionality."""

    def _get_common_patches(self, config):
        """Get common patches for initialization tests."""
        return [
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.load",
                return_value=config,
            ),
            patch("vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.QdrantVectorDB",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.create_llm",
                return_value=MagicMock(api_key="test-key"),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.AgenticRouter",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ChatGroq",
                return_value=MagicMock(),
            ),
        ]

    def test_initialization(self, qdrant_search_config):
        """Test pipeline initialization with valid config."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        patches = self._get_common_patches(qdrant_search_config)
        with (
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.generate",
                return_value="Final answer",
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ContextCompressor",
                return_value=MagicMock(),
            ),
        ):
            for p in patches:
                p.start()
            try:
                pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
                assert pipeline is not None
                assert pipeline.max_iterations == 3
                assert pipeline.compression_mode == "reranking"
                assert pipeline.collection_name == "test_agentic_rag"
            finally:
                for p in patches:
                    p.stop()

    def test_initialization_with_llm_extraction(self, qdrant_search_config):
        """Test initialization with llm_extraction compression mode."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        config = qdrant_search_config.copy()
        config["agentic"]["compression_mode"] = "llm_extraction"

        patches = self._get_common_patches(config)
        with (
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.generate",
                return_value="Final answer",
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ContextCompressor",
                return_value=MagicMock(),
            ),
        ):
            for p in patches:
                p.start()
            try:
                pipeline = QdrantAgenticRAGPipeline(config)
                assert pipeline.compression_mode == "llm_extraction"
            finally:
                for p in patches:
                    p.stop()

    def test_initialization_without_rag_llm(self, qdrant_search_config):
        """Test initialization fails when RAG LLM is not configured."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.load",
                return_value=qdrant_search_config,
            ),
            patch("vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.QdrantVectorDB",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.create_llm",
                return_value=None,
            ),
        ]

        for p in patches:
            p.start()
        try:
            with pytest.raises(ValueError, match="RAG LLM must be enabled"):
                QdrantAgenticRAGPipeline(qdrant_search_config)
        finally:
            for p in patches:
                p.stop()

    def _get_run_patches(
        self, config, mock_db, mock_router, mock_compressor, mock_rag_generate=None
    ):
        """Get patches for run method tests."""
        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.load",
                return_value=config,
            ),
            patch("vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.QdrantVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.create_llm",
                return_value=MagicMock(api_key="test-key"),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ContextCompressor",
                return_value=mock_compressor,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ChatGroq",
                return_value=MagicMock(),
            ),
        ]
        patches.append(
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.generate",
                mock_rag_generate or MagicMock(return_value="Generated answer"),
            )
        )
        return patches

    def test_run_with_search_action(
        self, qdrant_search_config, sample_search_documents
    ):
        """Test run method with search action."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_search_documents
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "action": "search",
            "reasoning": "Search for docs",
        }
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_search_documents[:3]

        patches = self._get_run_patches(
            qdrant_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
            result = pipeline.run("What are vector databases?")

            assert "final_answer" in result
            assert "documents" in result
            assert "intermediate_steps" in result
            mock_db.search.assert_called()
        finally:
            for p in patches:
                p.stop()

    def test_run_with_generate_action(
        self, qdrant_search_config, sample_search_documents
    ):
        """Test run method with generate action."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_search_documents
        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Search for docs"},
            {"action": "generate", "reasoning": "Generate answer"},
        ]
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_search_documents[:3]
        mock_rag_generate = MagicMock(return_value="Final answer")

        patches = self._get_run_patches(
            qdrant_search_config,
            mock_db,
            mock_router,
            mock_compressor,
            mock_rag_generate,
        )

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
            result = pipeline.run("What are vector databases?")

            assert "final_answer" in result
            mock_rag_generate.assert_called()
        finally:
            for p in patches:
                p.stop()

    def test_run_with_reflect_action(
        self, qdrant_search_config, sample_search_documents, sample_documents
    ):
        """Test run method with reflect action."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_search_documents
        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Search for docs"},
            {"action": "reflect", "reasoning": "Reflect on answer"},
            {"action": "generate", "reasoning": "Generate answer"},
        ]
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_documents[:3]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Reflection feedback")

        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.load",
                return_value=qdrant_search_config,
            ),
            patch("vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.QdrantVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.create_llm",
                return_value=mock_llm,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ContextCompressor",
                return_value=mock_compressor,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ChatGroq",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.generate",
                MagicMock(return_value="Generated answer"),
            ),
        ]

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
            result = pipeline.run("What are vector databases?")

            assert "final_answer" in result
        finally:
            for p in patches:
                p.stop()

    def test_run_with_custom_top_k(self, qdrant_search_config, sample_search_documents):
        """Test run method with custom top_k parameter."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_search_documents
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "action": "search",
            "reasoning": "Search for docs",
        }
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_search_documents[:3]

        patches = self._get_run_patches(
            qdrant_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
            pipeline.run("What are vector databases?", top_k=20)

            call_kwargs = mock_db.search.call_args
            assert call_kwargs[1]["top_k"] == 20
        finally:
            for p in patches:
                p.stop()

    def test_run_with_filters(self, qdrant_search_config, sample_search_documents):
        """Test run method with metadata filters."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_search_documents
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "action": "search",
            "reasoning": "Search for docs",
        }
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_search_documents[:3]

        patches = self._get_run_patches(
            qdrant_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
            filters = {"source": "docs"}
            pipeline.run("What are vector databases?", filters=filters)

            call_kwargs = mock_db.search.call_args
            assert call_kwargs[1]["filters"] == filters
        finally:
            for p in patches:
                p.stop()

    def test_run_with_empty_results(self, qdrant_search_config):
        """Test run method when no documents are retrieved."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = []
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "action": "search",
            "reasoning": "Search for docs",
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Fallback answer")

        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.load",
                return_value=qdrant_search_config,
            ),
            patch("vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.QdrantVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.create_llm",
                return_value=mock_llm,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ChatGroq",
                return_value=MagicMock(),
            ),
        ]

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
            result = pipeline.run("What are vector databases?")

            assert "final_answer" in result
            mock_llm.invoke.assert_called()
        finally:
            for p in patches:
                p.stop()

    def test_run_max_iterations_reached(self, qdrant_search_config):
        """Test run method when max iterations are reached without generation."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = []
        mock_router = MagicMock()
        mock_router.route.return_value = {
            "action": "search",
            "reasoning": "Keep searching",
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Fallback answer")

        config = qdrant_search_config.copy()
        config["agentic"]["max_iterations"] = 1

        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.load",
                return_value=config,
            ),
            patch("vectordb.langchain.agentic_rag.search.qdrant.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.QdrantVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RAGHelper.create_llm",
                return_value=mock_llm,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.qdrant.ChatGroq",
                return_value=MagicMock(),
            ),
        ]

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(config)
            result = pipeline.run("What are vector databases?")

            assert "final_answer" in result
        finally:
            for p in patches:
                p.stop()

    def test_run_intermediate_steps_tracking(
        self, qdrant_search_config, sample_search_documents
    ):
        """Test that intermediate steps are properly tracked."""
        from vectordb.langchain.agentic_rag.search.qdrant import (
            QdrantAgenticRAGPipeline,
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_search_documents
        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Search for docs"},
            {"action": "generate", "reasoning": "Generate answer"},
        ]
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_search_documents[:3]

        patches = self._get_run_patches(
            qdrant_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = QdrantAgenticRAGPipeline(qdrant_search_config)
            result = pipeline.run("What are vector databases?")

            assert "intermediate_steps" in result
            assert len(result["intermediate_steps"]) >= 1
            assert "reasoning" in result
        finally:
            for p in patches:
                p.stop()
