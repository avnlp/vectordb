"""Base test class for MMR (Maximal Marginal Relevance) pipeline tests.

This module provides a shared base class that encapsulates the common test logic
for all vector database backends (Chroma, Milvus, Pinecone, Qdrant, Weaviate).

Subclasses define only the database-specific configuration, patch targets, and
pipeline classes. All shared test methods live here.

Usage:
    class TestPineconeMMR(MmrTestBase):
        db_module = "pinecone"
        db_class_name = "PineconeVectorDB"
        integration_env_var = "PINECONE_API_KEY"
        indexing_pipeline_cls = PineconeMmrIndexingPipeline
        search_pipeline_cls = PineconeMmrSearchPipeline

        @property
        def unit_indexing_db_config(self) -> dict[str, Any]:
            return {"pinecone": {"api_key": "test-key", ...}}
        ...
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document


class MmrTestBase:
    """Base class for MMR pipeline tests across vector database backends.

    Subclasses must define these class attributes:
        db_module: Module name used in patch paths (e.g., "pinecone", "milvus").
        db_class_name: VectorDB class name to patch (e.g., "PineconeVectorDB").
        integration_env_var: Primary environment variable for integration tests.
        indexing_pipeline_cls: Indexing pipeline class to instantiate.
        search_pipeline_cls: Search pipeline class to instantiate.

    Subclasses must also define these properties returning the DB-specific
    portion of the config dict (keyed by db name, e.g. {"pinecone": {...}}):
        unit_indexing_db_config
        unit_search_db_config
        integration_indexing_db_config
        integration_search_db_config
    """

    db_module: str
    db_class_name: str
    integration_env_var: str
    indexing_pipeline_cls: type
    search_pipeline_cls: type

    @property
    def unit_indexing_db_config(self) -> dict[str, Any]:
        """Return database configuration for unit testing indexing pipelines."""
        raise NotImplementedError

    @property
    def unit_search_db_config(self) -> dict[str, Any]:
        """Return database configuration for unit testing search pipelines."""
        raise NotImplementedError

    @property
    def integration_indexing_db_config(self) -> dict[str, Any]:
        """Return database configuration for integration testing indexing pipelines."""
        raise NotImplementedError

    @property
    def integration_search_db_config(self) -> dict[str, Any]:
        """Return database configuration for integration testing search pipelines."""
        raise NotImplementedError

    def _make_indexing_unit_config(self) -> dict[str, Any]:
        return {
            **self.unit_indexing_db_config,
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"name": "triviaqa", "limit": 10},
        }

    def _make_search_unit_config(self) -> dict[str, Any]:
        return {
            **self.unit_search_db_config,
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"name": "triviaqa"},
            "mmr": {"lambda_threshold": 0.5},
            "rag": {
                "enabled": True,
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
            },
        }

    def _make_indexing_integration_config(self) -> dict[str, Any]:
        return {
            **self.integration_indexing_db_config,
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"name": "triviaqa", "limit": 5},
        }

    def _make_search_integration_config(self) -> dict[str, Any]:
        return {
            **self.integration_search_db_config,
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"name": "triviaqa"},
            "mmr": {"lambda_threshold": 0.5},
            "rag": {
                "enabled": True,
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
            },
        }

    def test_indexing_unit(self) -> None:
        """Unit test for the indexing pipeline with mocked dependencies.

        Verifies that the indexing pipeline correctly:
            - Loads documents via DataloaderCatalog
            - Generates embeddings via EmbedderFactory
            - Upserts documents to the vector database
            - Returns the count of indexed documents
        """
        base = f"vectordb.haystack.mmr.indexing.{self.db_module}"
        with (
            patch(f"{base}.DataloaderCatalog") as mock_dataloader_catalog,
            patch(f"{base}.{self.db_class_name}") as mock_db_cls,
            patch(f"{base}.EmbedderFactory") as mock_embedder_factory,
        ):
            sample_documents = [
                Document(content="Test doc 1", meta={"id": "1"}),
                Document(content="Test doc 2", meta={"id": "2"}),
            ]
            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = sample_documents
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_dataloader_catalog.create.return_value = mock_loader

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {
                "documents": [
                    Document(content="Test doc 1", embedding=[0.1] * 384),
                    Document(content="Test doc 2", embedding=[0.2] * 384),
                ]
            }
            mock_embedder_factory.create_document_embedder.return_value = mock_embedder

            mock_db = MagicMock()
            mock_db.upsert.return_value = 2
            mock_db_cls.return_value = mock_db

            pipeline = self.indexing_pipeline_cls(self._make_indexing_unit_config())
            result = pipeline.run()

            assert result["documents_indexed"] == 2
            mock_dataloader_catalog.create.assert_called_once()
            mock_embedder.run.assert_called_once()
            mock_db.upsert.assert_called_once()

    def test_search_unit(self) -> None:
        """Unit test for the search pipeline with mocked dependencies.

        Verifies that the search pipeline correctly:
            - Embeds queries via EmbedderFactory
            - Queries the vector database for candidate documents
            - Applies diversity reranking via RerankerFactory (MMR)
            - Generates answers via RAGHelper when enabled
        """
        base = f"vectordb.haystack.mmr.search.{self.db_module}"
        with (
            patch(f"{base}.RAGHelper") as mock_rag_helper,
            patch(f"{base}.RerankerFactory") as mock_ranker_factory,
            patch(f"{base}.{self.db_class_name}") as mock_db_cls,
            patch(f"{base}.EmbedderFactory") as mock_embedder_factory,
        ):
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_factory.create_text_embedder.return_value = mock_embedder

            candidates = [
                Document(content=f"Doc {i}", embedding=[0.1 * i] * 384)
                for i in range(10)
            ]
            mock_db = MagicMock()
            mock_db.query.return_value = candidates
            mock_db_cls.return_value = mock_db

            mock_ranker = MagicMock()
            mock_ranker.run.return_value = {"documents": candidates[:5]}
            mock_ranker_factory.create_diversity_ranker.return_value = mock_ranker

            mock_generator = MagicMock()
            mock_rag_helper.create_generator.return_value = mock_generator
            mock_rag_helper.generate.return_value = "Generated answer"

            pipeline = self.search_pipeline_cls(self._make_search_unit_config())
            result = pipeline.search("What is AI?", top_k=5, top_k_candidates=20)

            assert len(result["documents"]) == 5
            assert result["answer"] == "Generated answer"
            mock_ranker.run.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_indexing_integration(self) -> None:
        """Integration test for live document indexing.

        Tests the full indexing pipeline against a live database instance,
        including document loading, embedding generation, and vector upsert.

        Skips if the required environment variable is not set.
        """
        if not os.getenv(self.integration_env_var):
            pytest.skip(f"{self.integration_env_var} not set")

        pipeline = self.indexing_pipeline_cls(self._make_indexing_integration_config())
        result = pipeline.run()
        assert result["documents_indexed"] > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_search_with_rag_integration(self) -> None:
        """Integration test for live search with MMR and RAG generation.

        Tests the full search pipeline against live database and Groq APIs,
        including query embedding, vector search, MMR diversity ranking,
        and LLM-based answer generation.

        Skips if the required environment variables are not set.
        """
        if not os.getenv(self.integration_env_var) or not os.getenv("GROQ_API_KEY"):
            pytest.skip(f"{self.integration_env_var} or GROQ_API_KEY not set")

        pipeline = self.search_pipeline_cls(self._make_search_integration_config())
        result = pipeline.search("What is machine learning?", top_k=5)
        assert "documents" in result
        assert result["documents"] is not None
        assert "answer" in result
        assert result["answer"] is not None
        assert len(result["answer"]) > 0
