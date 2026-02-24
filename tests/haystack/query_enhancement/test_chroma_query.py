"""Unit tests for Chroma query enhancement pipelines."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.query_enhancement.indexing.chroma import (
    ChromaQueryEnhancementIndexingPipeline,
)
from vectordb.haystack.query_enhancement.search.chroma import (
    ChromaQueryEnhancementSearchPipeline,
)


class TestChromaQueryEnhancement:
    """Test class for Chroma query enhancement feature."""

    @pytest.fixture
    def chroma_config(self) -> dict[str, Any]:
        """Sample Chroma configuration."""
        return {
            "chroma": {
                "persist_directory": "/tmp/chroma_test",
                "collection_name": "test_collection",
            },
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"name": "triviaqa", "limit": 10},
            "query_enhancement": {
                "type": "multi_query",
                "num_queries": 3,
                "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
            },
            "rag": {
                "enabled": True,
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
            },
        }

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Sample documents for testing."""
        return [
            Document(content="Test document 1", embedding=[0.1] * 384),
            Document(content="Test document 2", embedding=[0.2] * 384),
            Document(content="Test document 3", embedding=[0.3] * 384),
        ]

    @patch("vectordb.haystack.query_enhancement.indexing.chroma.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.query_enhancement.indexing.chroma.create_document_embedder"
    )
    @patch("vectordb.haystack.query_enhancement.indexing.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.chroma.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
        chroma_config: dict[str, Any],
    ) -> None:
        """Test indexing pipeline initializes correctly."""
        mock_load_config.return_value = chroma_config

        pipeline = ChromaQueryEnhancementIndexingPipeline("/tmp/fake_config.yaml")

        mock_load_config.assert_called_once_with("/tmp/fake_config.yaml")
        mock_validate.assert_called_once_with(chroma_config)
        mock_dataloader_catalog.create.assert_called_once()
        mock_embedder_factory.assert_called_once_with(chroma_config)
        mock_db_cls.assert_called_once()
        assert pipeline.config == chroma_config

    @patch("vectordb.haystack.query_enhancement.indexing.chroma.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.query_enhancement.indexing.chroma.create_document_embedder"
    )
    @patch("vectordb.haystack.query_enhancement.indexing.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.chroma.load_config")
    def test_indexing_pipeline_run(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test indexing pipeline run method."""
        mock_load_config.return_value = chroma_config

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_dataloader_catalog.create.return_value = mock_loader

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db_cls.return_value = mock_db

        pipeline = ChromaQueryEnhancementIndexingPipeline("/tmp/fake_config.yaml")
        result = pipeline.run()

        assert result["documents_indexed"] == 3
        mock_loader.load.assert_called_once()
        mock_dataset.to_haystack.assert_called_once()
        mock_embedder.run.assert_called_once_with(documents=sample_documents)
        mock_db.upsert.assert_called_once_with(sample_documents)

    @patch("vectordb.haystack.query_enhancement.search.chroma.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.chroma.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.chroma.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.chroma.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer: MagicMock,
        mock_groq: MagicMock,
        chroma_config: dict[str, Any],
    ) -> None:
        """Test search pipeline initializes correctly."""
        mock_load_config.return_value = chroma_config

        pipeline = ChromaQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")

        mock_load_config.assert_called_once_with("/tmp/fake_config.yaml")
        mock_validate.assert_called_once_with(chroma_config)
        mock_embedder_factory.assert_called_once_with(chroma_config)
        mock_query_enhancer.assert_called_once()
        mock_db_cls.assert_called_once()
        assert pipeline.config == chroma_config

    @patch("vectordb.haystack.query_enhancement.search.chroma.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.chroma.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.chroma.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.chroma.load_config")
    def test_search_pipeline_run_multi_query(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test search pipeline with multi-query enhancement."""
        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = [
            "query 1",
            "query 2",
            "query 3",
        ]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_cls.return_value = mock_db

        mock_rag_gen = MagicMock()
        mock_rag_gen.run.return_value = {"replies": ["Generated answer"]}
        mock_groq.return_value = mock_rag_gen

        pipeline = ChromaQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        assert "answer" in result
        mock_query_enhancer.generate_multi_queries.assert_called_once()

    @patch("vectordb.haystack.query_enhancement.search.chroma.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.chroma.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.chroma.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.chroma.load_config")
    def test_search_pipeline_run_without_rag(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test search pipeline without RAG enabled."""
        config_no_rag = chroma_config.copy()
        config_no_rag["rag"] = {"enabled": False}
        mock_load_config.return_value = config_no_rag

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = ["query 1"]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_cls.return_value = mock_db

        pipeline = ChromaQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        assert "answer" not in result

    @patch("vectordb.haystack.query_enhancement.search.chroma.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.chroma.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.chroma.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.chroma.load_config")
    def test_search_pipeline_hyde_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test search pipeline with HyDE enhancement."""
        hyde_config = chroma_config.copy()
        hyde_config["query_enhancement"] = {
            "type": "hyde",
            "num_hyde_docs": 2,
            "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
        }
        hyde_config["rag"] = {"enabled": False}
        mock_load_config.return_value = hyde_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_cls.return_value = mock_db

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_hypothetical_documents.return_value = [
            "hypothetical doc 1",
            "hypothetical doc 2",
        ]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        pipeline = ChromaQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        mock_query_enhancer.generate_hypothetical_documents.assert_called_once_with(
            "test query", 2
        )

    @patch("vectordb.haystack.query_enhancement.search.chroma.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.chroma.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.chroma.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.chroma.load_config")
    def test_search_pipeline_step_back_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test search pipeline with step-back enhancement."""
        step_back_config = chroma_config.copy()
        step_back_config["query_enhancement"] = {
            "type": "step_back",
            "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
        }
        step_back_config["rag"] = {"enabled": False}
        mock_load_config.return_value = step_back_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_cls.return_value = mock_db

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_step_back_query.return_value = "step back query"
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        pipeline = ChromaQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        mock_query_enhancer.generate_step_back_query.assert_called_once_with(
            "test query"
        )

    @patch("vectordb.haystack.query_enhancement.search.chroma.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.chroma.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.chroma.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.chroma.load_config")
    def test_search_pipeline_handles_search_failure(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test search pipeline logs when a search task fails."""
        config_no_rag = chroma_config.copy()
        config_no_rag["rag"] = {"enabled": False}
        mock_load_config.return_value = config_no_rag

        mock_embedder_factory.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()
        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = ["good", "bad"]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        pipeline = ChromaQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        pipeline.logger = MagicMock()

        def search_side_effect(query: str, top_k: int) -> list[Document]:
            if query == "bad":
                raise RuntimeError("boom")
            return sample_documents[:1]

        pipeline._search_single_query = MagicMock(side_effect=search_side_effect)
        result = pipeline.run("test query", top_k=5)

        assert len(result["documents"]) == 1
        pipeline.logger.error.assert_called_once()
        pipeline._search_single_query.assert_any_call("good", 5)
        pipeline._search_single_query.assert_any_call("bad", 5)

    @patch("vectordb.haystack.query_enhancement.search.chroma.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.chroma.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.chroma.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.chroma.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.chroma.load_config")
    def test_search_pipeline_handles_rag_failure(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test search pipeline returns documents when RAG generation fails."""
        mock_load_config.return_value = chroma_config

        mock_embedder_factory.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()
        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = ["query 1"]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        mock_generator = MagicMock()
        mock_generator.run.side_effect = RuntimeError("RAG failed")
        mock_groq.return_value = mock_generator

        pipeline = ChromaQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        pipeline.logger = MagicMock()
        pipeline._search_single_query = MagicMock(return_value=sample_documents[:1])

        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        assert "answer" not in result
        pipeline.logger.error.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("CHROMA_PERSIST_DIR"),
        reason="CHROMA_PERSIST_DIR not set for integration test",
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Chroma indexing pipeline."""
        pytest.skip("Integration test placeholder.")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("CHROMA_PERSIST_DIR"),
        reason="CHROMA_PERSIST_DIR not set for integration test",
    )
    def test_search_integration(self) -> None:
        """Integration test for Chroma search pipeline."""
        pytest.skip("Integration test placeholder.")
