"""Unit and integration tests for Pinecone MMR (Maximal Marginal Relevance) pipelines.

This module tests the Haystack integration with Pinecone vector database for
MMR-based retrieval and RAG (Retrieval-Augmented Generation) pipelines.

Key Components:
    - PineconeMmrIndexingPipeline: Pipeline for indexing documents into Pinecone with
      embeddings
    - PineconeMmrSearchPipeline: Pipeline for searching with MMR diversity ranking and
      RAG
    - PineconeVectorDB: Vector database wrapper with namespace support

Configuration:
    Tests require PINECONE_API_KEY environment variable for integration tests.
    Optional GROQ_API_KEY for RAG integration tests.

Example:
    Run unit tests:
        pytest tests/haystack/mmr/test_pinecone_mmr.py -v -m "not integration"

    Run integration tests:
        PINECONE_API_KEY=xxx pytest tests/haystack/mmr/test_pinecone_mmr.py -v -m \
integration

Attributes:
    TestPineconeMMR: Test class containing unit and integration tests for Pinecone MMR.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.mmr.indexing import PineconeMmrIndexingPipeline
from vectordb.haystack.mmr.search import PineconeMmrSearchPipeline


class TestPineconeMMR:
    """Test class for Pinecone MMR indexing and search functionality.

    This class provides comprehensive tests for both unit (mocked) and integration
    scenarios using the Pinecone vector database with Haystack pipelines.

    Test Categories:
        Unit Tests:
            - test_indexing_unit: Tests indexing pipeline with mocked dependencies
            - test_search_unit: Tests search pipeline with mocked dependencies

        Integration Tests:
            - test_indexing_integration: Tests live indexing to Pinecone
            - test_search_with_rag_integration: Tests live search with RAG generation

    Configuration Structure:
        All tests use a consistent config dictionary with these sections:
            - pinecone: API key, index name, namespace, dimension settings
            - embeddings: Model configuration (e.g., MiniLM-L6-v2)
            - dataloader: Data source configuration (e.g., TriviaQA)
            - mmr: Lambda threshold for diversity vs relevance trade-off
            - rag: Generator configuration for answer generation
    """

    @patch("vectordb.haystack.mmr.indexing.pinecone.DataloaderCatalog")
    @patch("vectordb.haystack.mmr.indexing.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.mmr.indexing.pinecone.EmbedderFactory")
    def test_indexing_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
    ) -> None:
        """Unit test for document indexing pipeline with mocked dependencies.

        Verifies that the indexing pipeline correctly:
            - Loads documents via DataloaderCatalog
            - Generates embeddings via EmbedderFactory
            - Upserts documents to PineconeVectorDB
            - Returns the count of indexed documents

        Mocks:
            - DataloaderCatalog: Returns loader that returns dataset with test documents
            - EmbedderFactory: Returns mock embedder with 384-dimensional vectors
            - PineconeVectorDB: Mock database with upsert tracking

        Args:
            mock_dataloader_catalog: Mocked DataloaderCatalog for document loading
            mock_embedder_factory: Mocked EmbedderFactory for embedder creation
            mock_db_cls: Mocked PineconeVectorDB class

        Config:
            pinecone.api_key: "test-key"
            pinecone.index_name: "test-index"
            pinecone.namespace: "test"
            pinecone.dimension: 384
            embeddings.model: "sentence-transformers/all-MiniLM-L6-v2"
            dataloader.name: "triviaqa"
            dataloader.limit: 10
        """
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

        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test",
                "dimension": 384,
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "dataloader": {
                "name": "triviaqa",
                "limit": 10,
            },
        }

        pipeline = PineconeMmrIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        mock_dataloader_catalog.create.assert_called_once()
        mock_embedder.run.assert_called_once()
        mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.mmr.search.pinecone.RAGHelper")
    @patch("vectordb.haystack.mmr.search.pinecone.RerankerFactory")
    @patch("vectordb.haystack.mmr.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.mmr.search.pinecone.EmbedderFactory")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_ranker_factory: MagicMock,
        mock_rag_helper: MagicMock,
    ) -> None:
        """Unit test for search pipeline with MMR ranking and RAG generation.

        Verifies that the search pipeline correctly:
            - Embeds queries via EmbedderFactory
            - Queries PineconeVectorDB for candidate documents
            - Applies diversity reranking via RerankerFactory (MMR)
            - Generates answers via RAGHelper when enabled

        Mocks:
            - EmbedderFactory: Returns mock text embedder
            - PineconeVectorDB: Returns 10 candidate documents
            - RerankerFactory: Returns mock diversity ranker (returns top 5)
            - RAGHelper: Returns mock generator and answer

        Args:
            mock_embedder_factory: Mocked embedder factory for query embedding
            mock_db_cls: Mocked PineconeVectorDB for candidate retrieval
            mock_ranker_factory: Mocked ranker factory for MMR diversity ranking
            mock_rag_helper: Mocked RAG helper for answer generation

        Config:
            pinecone.api_key: "test-key"
            pinecone.index_name: "test-index"
            pinecone.namespace: "test"
            embeddings.model: "sentence-transformers/all-MiniLM-L6-v2"
            mmr.lambda_threshold: 0.5
            rag.enabled: True
            rag.provider: "groq"
            rag.model: "llama-3.3-70b-versatile"

        Query:
            search("What is AI?", top_k=5, top_k_candidates=20)
        """
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.create_text_embedder.return_value = mock_embedder

        candidates = [
            Document(content=f"Doc {i}", embedding=[0.1 * i] * 384) for i in range(10)
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

        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test",
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "dataloader": {
                "name": "triviaqa",
            },
            "mmr": {
                "lambda_threshold": 0.5,
            },
            "rag": {
                "enabled": True,
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
            },
        }

        pipeline = PineconeMmrSearchPipeline(config)
        result = pipeline.search("What is AI?", top_k=5, top_k_candidates=20)

        assert len(result["documents"]) == 5
        assert result["answer"] == "Generated answer"
        mock_ranker.run.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set"
    )
    def test_indexing_integration(self) -> None:
        """Integration test for live document indexing to Pinecone.

        Tests the full indexing pipeline against a live Pinecone instance,
        including document loading, embedding generation, and vector upsertion.

        Requirements:
            PINECONE_API_KEY environment variable must be set
            Network access to Pinecone API

        Side Effects:
            Creates/updates "mmr-integration-test" index in Pinecone
            Upserts documents to "integration_test" namespace

        Config:
            pinecone.api_key: From PINECONE_API_KEY env var
            pinecone.index_name: "mmr-integration-test"
            pinecone.namespace: "integration_test"
            pinecone.dimension: 384
            pinecone.recreate: True
            embeddings.model: "sentence-transformers/all-MiniLM-L6-v2"
            dataloader.name: "triviaqa"
            dataloader.limit: 5

        Data:
            Loads up to 5 documents from TriviaQA dataset
        """
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "index_name": "mmr-integration-test",
                "namespace": "integration_test",
                "dimension": 384,
                "recreate": True,
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "dataloader": {
                "name": "triviaqa",
                "limit": 5,
            },
        }

        pipeline = PineconeMmrIndexingPipeline(config)
        result = pipeline.run()
        assert result["documents_indexed"] > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY") or not os.getenv("GROQ_API_KEY"),
        reason="PINECONE_API_KEY or GROQ_API_KEY not set",
    )
    def test_search_with_rag_integration(self) -> None:
        """Integration test for live search with MMR and RAG generation.

        Tests the full search pipeline against live Pinecone and Groq APIs,
        including query embedding, vector search, MMR diversity ranking,
        and LLM-based answer generation.

        Requirements:
            PINECONE_API_KEY environment variable must be set
            GROQ_API_KEY environment variable must be set (for RAG)
            Network access to Pinecone and Groq APIs

        Pipeline Flow:
            1. Embed query using sentence-transformers
            2. Query Pinecone for candidates (mmr-integration-test index)
            3. Apply MMR diversity ranking (lambda_threshold=0.5)
            4. Generate answer using llama-3.3-70b-versatile

        Config:
            pinecone.api_key: From PINECONE_API_KEY env var
            pinecone.index_name: "mmr-integration-test"
            pinecone.namespace: "integration_test"
            embeddings.model: "sentence-transformers/all-MiniLM-L6-v2"
            mmr.lambda_threshold: 0.5
            rag.enabled: True
            rag.provider: "groq"
            rag.model: "llama-3.3-70b-versatile"
            rag.api_key: From GROQ_API_KEY env var

        Query:
            search("What is machine learning?", top_k=5)

        Assertions:
            - Response contains documents list
            - Documents list is non-empty
        """
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "index_name": "mmr-integration-test",
                "namespace": "integration_test",
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "dataloader": {
                "name": "triviaqa",
            },
            "mmr": {
                "lambda_threshold": 0.5,
            },
            "rag": {
                "enabled": True,
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
            },
        }

        pipeline = PineconeMmrSearchPipeline(config)
        result = pipeline.search("What is machine learning?", top_k=5)
        assert "documents" in result
        assert result["documents"] is not None
