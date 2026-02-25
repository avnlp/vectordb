"""Tests for Milvus agentic RAG search pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest


class TestMilvusAgenticRAGSearch:
    """Tests for MilvusAgenticRAGPipeline search functionality."""

    def _get_common_patches(self, config):
        """Get common patches for initialization tests."""
        return [
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.load",
                return_value=config,
            ),
            patch("vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.MilvusVectorDB",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.create_llm",
                return_value=MagicMock(api_key="test-key"),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.AgenticRouter",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ChatGroq",
                return_value=MagicMock(),
            ),
        ]

    def test_initialization(self, milvus_search_config):
        """Test pipeline initialization with valid config."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
        )

        patches = self._get_common_patches(milvus_search_config)
        with (
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.generate",
                return_value="Final answer",
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ContextCompressor",
                return_value=MagicMock(),
            ),
        ):
            for p in patches:
                p.start()
            try:
                pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
                assert pipeline is not None
                assert pipeline.max_iterations == 3
                assert pipeline.compression_mode == "reranking"
                assert pipeline.collection_name == "test_agentic_rag"
            finally:
                for p in patches:
                    p.stop()

    def test_initialization_with_llm_extraction(self, milvus_search_config):
        """Test initialization with llm_extraction compression mode."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
        )

        config = milvus_search_config.copy()
        config["agentic"]["compression_mode"] = "llm_extraction"

        patches = self._get_common_patches(config)
        with (
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.generate",
                return_value="Final answer",
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ContextCompressor",
                return_value=MagicMock(),
            ),
        ):
            for p in patches:
                p.start()
            try:
                pipeline = MilvusAgenticRAGPipeline(config)
                assert pipeline.compression_mode == "llm_extraction"
            finally:
                for p in patches:
                    p.stop()

    def test_initialization_without_rag_llm(self, milvus_search_config):
        """Test initialization fails when RAG LLM is not configured."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
        )

        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.load",
                return_value=milvus_search_config,
            ),
            patch("vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.MilvusVectorDB",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.create_llm",
                return_value=None,
            ),
        ]

        for p in patches:
            p.start()
        try:
            with pytest.raises(ValueError, match="RAG LLM must be enabled"):
                MilvusAgenticRAGPipeline(milvus_search_config)
        finally:
            for p in patches:
                p.stop()

    def _get_run_patches(
        self, config, mock_db, mock_router, mock_compressor, mock_rag_generate=None
    ):
        """Get patches for run method tests."""
        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.load",
                return_value=config,
            ),
            patch("vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.MilvusVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.create_llm",
                return_value=MagicMock(api_key="test-key"),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ContextCompressor",
                return_value=mock_compressor,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ChatGroq",
                return_value=MagicMock(),
            ),
        ]
        patches.append(
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.generate",
                mock_rag_generate or MagicMock(return_value="Generated answer"),
            )
        )
        return patches

    def test_run_with_search_action(
        self, milvus_search_config, sample_search_documents
    ):
        """Test run method with search action."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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
            milvus_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
            result = pipeline.run("What is Python?")

            assert "final_answer" in result
            assert "documents" in result
            assert "intermediate_steps" in result
            mock_db.search.assert_called()
        finally:
            for p in patches:
                p.stop()

    def test_run_with_generate_action(
        self, milvus_search_config, sample_search_documents
    ):
        """Test run method with generate action."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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
            milvus_search_config,
            mock_db,
            mock_router,
            mock_compressor,
            mock_rag_generate,
        )

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
            result = pipeline.run("What is Python?")

            assert "final_answer" in result
            mock_rag_generate.assert_called()
        finally:
            for p in patches:
                p.stop()

    def test_run_with_reflect_action(
        self, milvus_search_config, sample_search_documents, sample_documents
    ):
        """Test run method with reflect action."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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
                "vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.load",
                return_value=milvus_search_config,
            ),
            patch("vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.MilvusVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.create_llm",
                return_value=mock_llm,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ContextCompressor",
                return_value=mock_compressor,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ChatGroq",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.generate",
                MagicMock(return_value="Generated answer"),
            ),
        ]

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
            result = pipeline.run("What is Python?")

            assert "final_answer" in result
        finally:
            for p in patches:
                p.stop()

    def test_run_with_custom_top_k(self, milvus_search_config, sample_search_documents):
        """Test run method with custom top_k parameter."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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
            milvus_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
            pipeline.run("What is Python?", top_k=20)

            call_kwargs = mock_db.search.call_args
            assert call_kwargs[1]["top_k"] == 20
        finally:
            for p in patches:
                p.stop()

    def test_run_with_filters(self, milvus_search_config, sample_search_documents):
        """Test run method with metadata filters."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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
            milvus_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
            filters = {"source": "wiki"}
            pipeline.run("What is Python?", filters=filters)

            call_kwargs = mock_db.search.call_args
            assert call_kwargs[1]["filters"] == filters
        finally:
            for p in patches:
                p.stop()

    def test_run_with_empty_results(self, milvus_search_config):
        """Test run method when no documents are retrieved."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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
                "vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.load",
                return_value=milvus_search_config,
            ),
            patch("vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.MilvusVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.create_llm",
                return_value=mock_llm,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ChatGroq",
                return_value=MagicMock(),
            ),
        ]

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
            result = pipeline.run("What is Python?")

            assert "final_answer" in result
            mock_llm.invoke.assert_called()
        finally:
            for p in patches:
                p.stop()

    def test_run_max_iterations_reached(self, milvus_search_config):
        """Test run method when max iterations are reached without generation."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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

        config = milvus_search_config.copy()
        config["agentic"]["max_iterations"] = 1

        patches = [
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.load",
                return_value=config,
            ),
            patch("vectordb.langchain.agentic_rag.search.milvus.ConfigLoader.validate"),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.create_embedder",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.MilvusVectorDB",
                return_value=mock_db,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RAGHelper.create_llm",
                return_value=mock_llm,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.RerankerHelper.create_reranker",
                return_value=MagicMock(),
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.AgenticRouter",
                return_value=mock_router,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.EmbedderHelper.embed_query",
                return_value=[0.1] * 384,
            ),
            patch(
                "vectordb.langchain.agentic_rag.search.milvus.ChatGroq",
                return_value=MagicMock(),
            ),
        ]

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            assert "final_answer" in result
        finally:
            for p in patches:
                p.stop()

    def test_run_intermediate_steps_tracking(
        self, milvus_search_config, sample_search_documents
    ):
        """Test that intermediate steps are properly tracked."""
        from vectordb.langchain.agentic_rag.search.milvus import (
            MilvusAgenticRAGPipeline,
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
            milvus_search_config, mock_db, mock_router, mock_compressor
        )

        for p in patches:
            p.start()
        try:
            pipeline = MilvusAgenticRAGPipeline(milvus_search_config)
            result = pipeline.run("What is Python?")

            assert "intermediate_steps" in result
            assert len(result["intermediate_steps"]) >= 1
            assert "reasoning" in result
        finally:
            for p in patches:
                p.stop()
