"""Unit and integration tests for Chroma MMR (Maximal Marginal Relevance) pipelines.

This module provides comprehensive test coverage for the Haystack-based Chroma
integration with MMR reranking capabilities. MMR enables diverse document
retrieval by balancing relevance against redundancy.

Tested Components:
    - ChromaMmrIndexingPipeline: Pipeline for indexing documents with embeddings
      into local Chroma vector storage
    - ChromaMmrSearchPipeline: Pipeline for searching with MMR reranking and
      optional RAG (Retrieval-Augmented Generation) support

Chroma Configuration:
    Tests use ChromaVectorDB with local persist_directory for isolated,
    file-based vector storage. The persist_directory ensures vectors are
    stored locally rather than in-memory, enabling persistence across sessions.

MMR Configuration:
    MMR reranking uses a lambda_threshold parameter (0.0-1.0) to control the
    trade-off between relevance and diversity:
    - lambda=0.0: Maximum diversity
    - lambda=1.0: Maximum relevance
    - lambda=0.5: Balanced approach (default in tests)

Integration Tests:
    Tests marked with @pytest.mark.integration require:
    - CHROMA_HOST environment variable set
    - GROQ_API_KEY for RAG integration tests
    These tests validate real Chroma connections and LLM generation.
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.mmr.indexing import ChromaMmrIndexingPipeline
from vectordb.haystack.mmr.search import ChromaMmrSearchPipeline


class TestChromaMMR:
    """Test suite for Chroma MMR indexing and search pipelines.

    This class covers both unit tests with mocked dependencies and integration
    tests with real Chroma connections. The MMR approach ensures diverse
    results by reranking initial retrieval candidates.

    Test Categories:
        Unit Tests: Mock all external dependencies (ChromaVectorDB, embedders,
            rerankers) for fast, isolated testing
        Integration Tests: Require live Chroma instance and optionally
            GROQ API for end-to-end RAG validation

    Configuration Pattern:
        Tests use a nested config dict with keys for:
        - chroma: persist_directory, collection_name, dimension, recreate
        - embeddings: model specification
        - dataloader: dataset and sample limits
        - mmr: lambda_threshold for diversity tuning
        - rag: provider, model, API credentials
    """

    @patch("vectordb.haystack.mmr.indexing.chroma.DataloaderCatalog")
    @patch("vectordb.haystack.mmr.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.mmr.indexing.chroma.EmbedderFactory")
    def test_indexing_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
    ) -> None:
        """Unit test verifying document indexing pipeline with mocked dependencies.

        Validates that ChromaMmrIndexingPipeline correctly:
        1. Loads documents via DataloaderCatalog with configured limit
        2. Generates embeddings using EmbedderFactory document embedder
        3. Upserts documents into ChromaVectorDB local storage
        4. Returns accurate count of indexed documents

        Mock Setup:
            - DataloaderCatalog returns loader with sample docs
            - Embedder generates 384-dimensional vectors (MiniLM-L6-v2 size)
            - ChromaVectorDB upsert operation succeeds

        Expected Result:
            Result dict contains documents_indexed=2 confirming successful
            indexing of all loaded documents.
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
            "chroma": {
                "persist_directory": "/tmp/chroma_test",
                "collection_name": "test-collection",
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

        pipeline = ChromaMmrIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        mock_dataloader_catalog.create.assert_called_once()
        mock_embedder.run.assert_called_once()
        mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.mmr.search.chroma.RAGHelper")
    @patch("vectordb.haystack.mmr.search.chroma.RerankerFactory")
    @patch("vectordb.haystack.mmr.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.mmr.search.chroma.EmbedderFactory")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_ranker_factory: MagicMock,
        mock_rag_helper: MagicMock,
    ) -> None:
        """Unit test verifying search pipeline with MMR reranking and RAG.

        Validates that ChromaMmrSearchPipeline correctly:
        1. Embeds query using text embedder
        2. Retrieves candidate documents from ChromaVectorDB
        3. Applies MMR reranking via RerankerFactory diversity ranker
        4. Generates answers via RAGHelper when RAG is enabled

        Mock Setup:
            - EmbedderFactory creates text embedder returning 384-dim query vector
            - ChromaVectorDB returns 10 candidate documents with embeddings
            - RerankerFactory creates diversity ranker selecting top 5 diverse docs
            - RAGHelper creates generator and produces "Generated answer"

        Configuration:
            Uses lambda_threshold=0.5 for balanced relevance-diversity trade-off
            and enables RAG with Groq provider and llama-3.3-70b-versatile model.

        Expected Result:
            Result dict contains exactly 5 reranked documents and the generated
            RAG answer string.
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
            "chroma": {
                "persist_directory": "/tmp/chroma_test",
                "collection_name": "test-collection",
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

        pipeline = ChromaMmrSearchPipeline(config)
        result = pipeline.search("What is AI?", top_k=5, top_k_candidates=20)

        assert len(result["documents"]) == 5
        assert result["answer"] == "Generated answer"
        mock_ranker.run.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(not os.getenv("CHROMA_HOST"), reason="CHROMA_HOST not set")
    def test_indexing_integration(self) -> None:
        """Integration test validating live Chroma indexing pipeline.

        Tests end-to-end document indexing against a real Chroma instance:
        1. Connects to Chroma using CHROMA_HOST environment variable
        2. Loads TriviaQA dataset with sample limit of 5 documents
        3. Generates embeddings using sentence-transformers/all-MiniLM-L6-v2
        4. Stores vectors in local persist_directory (/tmp/chroma_integration_test)
        5. Creates or recreates collection "mmr-integration-test"

        Prerequisites:
            CHROMA_HOST environment variable must be set to a reachable
            Chroma server address (e.g., http://localhost:8000).

        Expected Result:
            Pipeline successfully indexes documents and returns
            documents_indexed > 0, confirming live Chroma connectivity.
        """
        config: dict[str, Any] = {
            "chroma": {
                "persist_directory": "/tmp/chroma_integration_test",
                "collection_name": "mmr-integration-test",
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

        pipeline = ChromaMmrIndexingPipeline(config)
        result = pipeline.run()
        assert result["documents_indexed"] > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("CHROMA_HOST") or not os.getenv("GROQ_API_KEY"),
        reason="CHROMA_HOST or GROQ_API_KEY not set",
    )
    def test_search_with_rag_integration(self) -> None:
        """Integration test validating live search with MMR reranking and RAG.

        Tests end-to-end search pipeline with real Chroma, MMR, and LLM:
        1. Connects to live Chroma instance using CHROMA_HOST
        2. Embeds query "What is machine learning?" using MiniLM-L6-v2
        3. Retrieves candidates from pre-populated mmr-integration-test collection
        4. Applies MMR reranking with lambda_threshold=0.5 for diversity
        5. Generates answer using Groq LLM (llama-3.3-70b-versatile)

        Prerequisites:
            - CHROMA_HOST: Points to running Chroma server
            - GROQ_API_KEY: Valid API key for Groq LLM access
            - Pre-indexed documents in mmr-integration-test collection

        RAG Configuration:
            Provider: groq
            Model: llama-3.3-70b-versatile (70B parameter Llama 3.3)
            API key: Loaded from GROQ_API_KEY environment variable

        Expected Result:
            Result dict contains both reranked documents list and a generated
            answer string from the LLM.
        """
        config: dict[str, Any] = {
            "chroma": {
                "persist_directory": "/tmp/chroma_integration_test",
                "collection_name": "mmr-integration-test",
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

        pipeline = ChromaMmrSearchPipeline(config)
        result = pipeline.search("What is machine learning?", top_k=5)
        assert "documents" in result
        assert result["documents"] is not None
