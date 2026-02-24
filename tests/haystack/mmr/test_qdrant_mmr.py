"""Test suite for Haystack Qdrant MMR (Maximal Marginal Relevance) pipelines.

This module provides comprehensive test coverage for the Qdrant MMR indexing and search
functionality using the Haystack 2.0 framework. It tests both unit and integration
scenarios for document indexing and diversity-based semantic search.

Key Components Tested:
    - QdrantMmrIndexingPipeline: Indexing pipeline using QdrantVectorDB for vector
      storage and EmbedderFactory for document embeddings. Uses DataLoaderHelper
      to load documents from supported datasets.
    - QdrantMmrSearchPipeline: Search pipeline supporting MMR diversity reranking
      via RerankerFactory and answer generation via RAGHelper. Performs vector
      search with query embedding and reranks for result diversity.

Dependencies:
    - Haystack framework for pipeline orchestration and Document handling
    - QdrantVectorDB for vector storage and similarity search
    - EmbedderFactory for creating text and document embedders
    - DataLoaderHelper for loading datasets (TriviaQA, ARC, PopQA, etc.)
    - RerankerFactory for MMR-based diversity ranking
    - RAGHelper for LLM-based answer generation

Test Coverage:
    - Unit tests with mocked dependencies for isolation
    - Integration tests requiring QDRANT_URL environment variable
    - Full end-to-end RAG integration tests requiring both QDRANT_URL and GROQ_API_KEY

Environment Variables:
    QDRANT_URL: Qdrant server URL for integration tests
    QDRANT_API_KEY: API key for Qdrant authentication
    GROQ_API_KEY: API key for Groq LLM provider (RAG tests only)
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.mmr.indexing import QdrantMmrIndexingPipeline
from vectordb.haystack.mmr.search import QdrantMmrSearchPipeline


class TestQdrantMMR:
    """Test suite for Qdrant MMR indexing and search pipelines.

    This test class validates both the indexing and search functionality of the
    Qdrant MMR pipelines, covering unit tests with mocked dependencies and
    integration tests against live Qdrant instances.

    The MMR (Maximal Marginal Relevance) algorithm balances relevance and diversity
    in search results by reranking candidate documents. Lambda threshold controls
    the trade-off between similarity to query and diversity from already-selected
    documents.

    Test Methods:
        test_indexing_unit: Validates indexing pipeline with mocked dependencies.
        test_search_unit: Validates search pipeline with mocked dependencies.
        test_indexing_integration: End-to-end indexing test against live Qdrant.
        test_search_with_rag_integration: End-to-end search with RAG against live
            Qdrant.

    Attributes:
        None - All dependencies are injected via mocks or environment variables.
    """

    @patch("vectordb.haystack.mmr.indexing.qdrant.DataloaderCatalog")
    @patch("vectordb.haystack.mmr.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.mmr.indexing.qdrant.EmbedderFactory")
    def test_indexing_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
    ) -> None:
        """Unit test for the Qdrant MMR indexing pipeline.

        Validates the indexing pipeline flow with mocked dependencies to ensure
        isolation from external services. Tests document loading, embedding generation,
        and vector storage without requiring a live Qdrant instance.

        Pipeline Flow Tested:
            1. DataloaderCatalog loads documents from configured dataset
            2. EmbedderFactory creates embedder and generates 384-dimensional vectors
            3. QdrantVectorDB stores embedded documents in collection
            4. Returns count of indexed documents

        Args:
            mock_dataloader_catalog: Mocked DataloaderCatalog for loading test docs.
            mock_embedder_factory: Mocked EmbedderFactory for creating document
                embedder.
            mock_db_cls: Mocked QdrantVectorDB class for vector storage.

        Asserts:
            documents_indexed equals 2 (matching mocked document count).
            All mocked methods called exactly once (upsert, embedder run, data loader).
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
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
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

        pipeline = QdrantMmrIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        mock_dataloader_catalog.create.assert_called_once()
        mock_embedder.run.assert_called_once()
        mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.mmr.search.qdrant.RAGHelper")
    @patch("vectordb.haystack.mmr.search.qdrant.RerankerFactory")
    @patch("vectordb.haystack.mmr.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.mmr.search.qdrant.EmbedderFactory")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_ranker_factory: MagicMock,
        mock_rag_helper: MagicMock,
    ) -> None:
        """Unit test for the Qdrant MMR search pipeline.

        Validates the search pipeline flow with mocked dependencies, including query
        embedding, vector retrieval, MMR diversity reranking, and RAG answer generation.
        Tests the full search flow without requiring live services.

        Pipeline Flow Tested:
            1. EmbedderFactory creates text embedder for query vectorization
            2. QdrantVectorDB retrieves top-k candidate documents by vector similarity
            3. RerankerFactory applies MMR diversity reranking (lambda_threshold: 0.5)
            4. RAGHelper generates contextual answer from reranked documents

        Args:
            mock_embedder_factory: Mocked EmbedderFactory for query embedding.
            mock_db_cls: Mocked QdrantVectorDB class for candidate retrieval.
            mock_ranker_factory: Mocked RerankerFactory for MMR diversity ranking.
            mock_rag_helper: Mocked RAGHelper for LLM answer generation.

        Asserts:
            Result contains 5 documents after MMR reranking (top_k=5).
            Generated answer matches expected mocked response.
            Diversity ranker invoked to rerank initial candidates.
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
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
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

        pipeline = QdrantMmrSearchPipeline(config)
        result = pipeline.search("What is AI?", top_k=5, top_k_candidates=20)

        assert len(result["documents"]) == 5
        assert result["answer"] == "Generated answer"
        mock_ranker.run.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="QDRANT_URL not set")
    def test_indexing_integration(self) -> None:
        """Integration test for Qdrant indexing with live vector database.

        Performs end-to-end document indexing against a live Qdrant instance to validate
        real-world pipeline behavior. Uses the TriviaQA dataloader to fetch documents,
        generates embeddings using sentence-transformers, and stores vectors in Qdrant.

        Requirements:
            QDRANT_URL environment variable must be set to the Qdrant server URL.
            QDRANT_API_KEY is optional for authenticated instances.

        Configuration:
            Collection: mmr_integration_test (recreate=True for clean state)
            Dimension: 384 (matching MiniLM-L6-v2 output)
            Dataset: TriviaQA with 5 document limit
            Model: sentence-transformers/all-MiniLM-L6-v2

        Asserts:
            At least one document successfully indexed (documents_indexed > 0).
        """
        config: dict[str, Any] = {
            "qdrant": {
                "url": os.getenv("QDRANT_URL"),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "collection_name": "mmr_integration_test",
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

        pipeline = QdrantMmrIndexingPipeline(config)
        result = pipeline.run()
        assert result["documents_indexed"] > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL") or not os.getenv("GROQ_API_KEY"),
        reason="QDRANT_URL or GROQ_API_KEY not set",
    )
    def test_search_with_rag_integration(self) -> None:
        """Integration test for Qdrant MMR search with live RAG answer generation.

        Performs end-to-end search with Maximal Marginal Relevance reranking and
        LLM-based answer generation against live Qdrant and Groq services. Validates
        the complete retrieval-augmented generation flow in production-like conditions.

        Pipeline Flow:
            1. Query "What is machine learning?" embedded via MiniLM-L6-v2
            2. Qdrant retrieves candidates from mmr_integration_test collection
            3. MMR reranking applied with lambda_threshold=0.5 for diversity
            4. Groq LLM (llama-3.3-70b-versatile) generates contextual answer

        Requirements:
            QDRANT_URL: Qdrant server URL for vector search
            QDRANT_API_KEY: Optional API key for Qdrant authentication
            GROQ_API_KEY: Required for LLM answer generation via Groq API

        Configuration:
            Collection: mmr_integration_test (must exist from prior indexing)
            MMR Lambda: 0.5 (balances relevance and diversity)
            LLM Provider: groq with llama-3.3-70b-versatile model
            Top-K: 5 documents returned after MMR reranking

        Asserts:
            Search result contains documents key with non-None value.
        """
        config: dict[str, Any] = {
            "qdrant": {
                "url": os.getenv("QDRANT_URL"),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "collection_name": "mmr_integration_test",
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

        pipeline = QdrantMmrSearchPipeline(config)
        result = pipeline.search("What is machine learning?", top_k=5)
        assert "documents" in result
        assert result["documents"] is not None
