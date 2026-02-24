"""Unit tests for Pinecone query enhancement pipelines."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.query_enhancement.indexing.pinecone import (
    PineconeQueryEnhancementIndexingPipeline,
)
from vectordb.haystack.query_enhancement.search.pinecone import (
    PineconeQueryEnhancementSearchPipeline,
)


class TestPineconeQueryEnhancement:
    """Test class for Pinecone query enhancement feature."""

    @pytest.fixture
    def pinecone_config(self) -> dict[str, Any]:
        """Sample Pinecone configuration for testing."""
        return {
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
            },
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"name": "triviaqa", "limit": 10},
            "query_enhancement": {
                "type": "multi_query",
                "num_queries": 3,
                "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
            },
            "rag": {"enabled": False},
        }

    @pytest.fixture
    def pinecone_config_with_rag(
        self, pinecone_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Configuration with RAG enabled."""
        config = pinecone_config.copy()
        config["rag"] = {
            "enabled": True,
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
        }
        return config

    @pytest.fixture
    def mock_documents(self) -> list[Document]:
        """Create mock documents with embeddings."""
        return [
            Document(content=f"Test document {i}", embedding=[0.1] * 384)
            for i in range(5)
        ]

    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.query_enhancement.indexing.pinecone.create_document_embedder"
    )
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
        pinecone_config: dict[str, Any],
    ) -> None:
        """Test that indexing pipeline initializes correctly."""
        mock_load_config.return_value = pinecone_config
        mock_dataset = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_dataloader_catalog.create.return_value = mock_loader
        mock_embedder = MagicMock()
        mock_embedder_factory.return_value = mock_embedder

        pipeline = PineconeQueryEnhancementIndexingPipeline("/tmp/fake_config.yaml")

        mock_load_config.assert_called_once_with("/tmp/fake_config.yaml")
        mock_validate.assert_called_once_with(pinecone_config)
        mock_dataloader_catalog.create.assert_called_once()
        mock_embedder_factory.assert_called_once_with(pinecone_config)
        assert pipeline.config == pinecone_config

    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.query_enhancement.indexing.pinecone.create_document_embedder"
    )
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.load_config")
    def test_indexing_pipeline_run(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
        pinecone_config: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test that indexing pipeline runs and indexes documents."""
        mock_load_config.return_value = pinecone_config

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = mock_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_dataloader_catalog.create.return_value = mock_loader

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": mock_documents}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.upsert.return_value = len(mock_documents)
        mock_db_cls.return_value = mock_db

        pipeline = PineconeQueryEnhancementIndexingPipeline("/tmp/fake_config.yaml")
        result = pipeline.run()

        assert result["documents_indexed"] == len(mock_documents)
        assert result["namespace"] == "test-namespace"
        mock_loader.load.assert_called_once()
        mock_dataset.to_haystack.assert_called_once()
        mock_embedder.run.assert_called_once_with(documents=mock_documents)
        mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.DataloaderCatalog")
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.query_enhancement.indexing.pinecone.create_document_embedder"
    )
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.indexing.pinecone.load_config")
    def test_indexing_pipeline_creates_index(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
        pinecone_config: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test that indexing pipeline creates index with correct dimension."""
        mock_load_config.return_value = pinecone_config

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = mock_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_dataloader_catalog.create.return_value = mock_loader

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": mock_documents}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.upsert.return_value = len(mock_documents)
        mock_db_cls.return_value = mock_db

        pipeline = PineconeQueryEnhancementIndexingPipeline("/tmp/fake_config.yaml")
        pipeline.run()

        mock_db.create_index.assert_called_once_with(dimension=384)

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config: dict[str, Any],
    ) -> None:
        """Test that search pipeline initializes correctly."""
        mock_load_config.return_value = pinecone_config

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")

        mock_load_config.assert_called_once_with("/tmp/fake_config.yaml")
        mock_validate.assert_called_once_with(pinecone_config)
        mock_embedder_factory.assert_called_once_with(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.rag_generator is None

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_multi_query(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with multi-query enhancement."""
        mock_load_config.return_value = pinecone_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = mock_documents
        mock_db_cls.return_value = mock_db

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = [
            "query 1",
            "query 2",
            "query 3",
        ]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        mock_query_enhancer.generate_multi_queries.assert_called_once_with(
            "test query", 3
        )

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_with_rag(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config_with_rag: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with RAG generation enabled."""
        mock_load_config.return_value = pinecone_config_with_rag

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = mock_documents
        mock_db_cls.return_value = mock_db

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = ["query 1"]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        mock_generator = MagicMock()
        mock_generator.run.return_value = {"replies": ["Generated answer"]}
        mock_groq.return_value = mock_generator

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        assert "answer" in result
        assert result["answer"] == "Generated answer"

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_hyde_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with HyDE enhancement."""
        hyde_config = pinecone_config.copy()
        hyde_config["query_enhancement"] = {
            "type": "hyde",
            "num_hyde_docs": 2,
            "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
        }
        mock_load_config.return_value = hyde_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = mock_documents
        mock_db_cls.return_value = mock_db

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_hypothetical_documents.return_value = [
            "hypothetical doc 1",
            "hypothetical doc 2",
        ]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        mock_query_enhancer.generate_hypothetical_documents.assert_called_once_with(
            "test query", 2
        )

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_step_back_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline with step-back enhancement."""
        step_back_config = pinecone_config.copy()
        step_back_config["query_enhancement"] = {
            "type": "step_back",
            "llm": {"model": "llama-3.3-70b-versatile", "api_key": "test-key"},
        }
        mock_load_config.return_value = step_back_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = mock_documents
        mock_db_cls.return_value = mock_db

        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_step_back_query.return_value = "step back query"
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        mock_query_enhancer.generate_step_back_query.assert_called_once_with(
            "test query"
        )

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_default_enhancement(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline default branch when type is unknown."""
        default_config = pinecone_config.copy()
        default_config["query_enhancement"] = {"type": "unknown"}
        mock_load_config.return_value = default_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = mock_documents
        mock_db_cls.return_value = mock_db

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        mock_db.query.assert_called_once()

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_handles_search_failure(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline logs when a search task fails."""
        config_no_rag = pinecone_config.copy()
        config_no_rag["rag"] = {"enabled": False}
        mock_load_config.return_value = config_no_rag

        mock_embedder_factory.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()
        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = ["good", "bad"]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        pipeline.logger = MagicMock()

        def search_side_effect(query: str, top_k: int) -> list[Document]:
            if query == "bad":
                raise RuntimeError("boom")
            return mock_documents[:1]

        pipeline._search_single_query = MagicMock(side_effect=search_side_effect)
        result = pipeline.run("test query", top_k=5)

        assert len(result["documents"]) == 1
        pipeline.logger.error.assert_called_once()
        pipeline._search_single_query.assert_any_call("good", 5)
        pipeline._search_single_query.assert_any_call("bad", 5)

    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_groq_generator")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.QueryEnhancer")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.create_text_embedder")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.validate_config")
    @patch("vectordb.haystack.query_enhancement.search.pinecone.load_config")
    def test_search_pipeline_handles_rag_failure(
        self,
        mock_load_config: MagicMock,
        mock_validate: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_query_enhancer_cls: MagicMock,
        mock_groq: MagicMock,
        pinecone_config_with_rag: dict[str, Any],
        mock_documents: list[Document],
    ) -> None:
        """Test search pipeline returns documents when RAG generation fails."""
        mock_load_config.return_value = pinecone_config_with_rag

        mock_embedder_factory.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()
        mock_query_enhancer = MagicMock()
        mock_query_enhancer.generate_multi_queries.return_value = ["query 1"]
        mock_query_enhancer_cls.return_value = mock_query_enhancer

        mock_generator = MagicMock()
        mock_generator.run.side_effect = RuntimeError("RAG failed")
        mock_groq.return_value = mock_generator

        pipeline = PineconeQueryEnhancementSearchPipeline("/tmp/fake_config.yaml")
        pipeline.logger = MagicMock()
        pipeline._search_single_query = MagicMock(return_value=mock_documents[:1])

        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        assert "answer" not in result
        pipeline.logger.error.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"),
        reason="PINECONE_API_KEY environment variable not set",
    )
    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY environment variable not set",
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Pinecone indexing pipeline."""
        pytest.skip("Integration test placeholder.")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"),
        reason="PINECONE_API_KEY environment variable not set",
    )
    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"),
        reason="GROQ_API_KEY environment variable not set",
    )
    def test_search_integration(self) -> None:
        """Integration test for Pinecone search pipeline."""
        pytest.skip("Integration test placeholder.")
