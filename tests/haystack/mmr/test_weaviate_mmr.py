"""Unit and integration tests for Weaviate MMR pipelines.

This module tests the Maximal Marginal Relevance (MMR) functionality within
the Haystack Weaviate integration. MMR provides a balance between relevance
and diversity in search results.

Tested Components:
    - WeaviateMmrIndexingPipeline: Indexing pipeline for MMR-enabled collections
    - WeaviateMmrSearchPipeline: Search pipeline with diversity reranking
    - WeaviateVectorDB: Vector database wrapper for Weaviate
    - MMR Diversity Ranker: Reranking component for diversity-aware retrieval
    - RAG Integration: Generative answering with MMR-diverse contexts

Test Categories:
    - Unit tests: Mock-based tests for pipeline logic and component interaction
    - Integration tests: End-to-end tests requiring live Weaviate and Groq services

Environment Variables:
    - WEAVIATE_URL: URL of the Weaviate instance (required for integration tests)
    - GROQ_API_KEY: API key for Groq LLM service (required for RAG integration tests)

Dependencies:
    - pytest: Testing framework with async and integration support
    - haystack: Document and pipeline abstractions
    - vectordb: Local package providing Weaviate and MMR implementations
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.mmr.indexing import WeaviateMmrIndexingPipeline
from vectordb.haystack.mmr.search import WeaviateMmrSearchPipeline


class TestWeaviateMMR:
    """Unit and integration tests for Weaviate MMR pipelines.

    This test class validates the functionality of the Weaviate MMR
    indexing and search pipelines. MMR (Maximal Marginal Relevance)
    optimizes for both query relevance and result diversity.

    Test Coverage:
        - Document indexing with embedding generation
        - MMR-based search with diversity reranking
        - RAG generation with diverse context windows
        - Configuration-driven pipeline initialization
        - Environment-based integration testing

    Attributes:
        None - Test class uses pytest fixtures and local configuration dictionaries.

    Example:
        Run unit tests only:
            pytest tests/haystack/mmr/test_weaviate_mmr.py -v

        Run with integration tests (requires WEAVIATE_URL):
            WEAVIATE_URL=http://localhost:8080 pytest \
tests/haystack/mmr/test_weaviate_mmr.py -v
    """

    @patch("vectordb.haystack.mmr.indexing.weaviate.DataloaderCatalog")
    @patch("vectordb.haystack.mmr.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.mmr.indexing.weaviate.EmbedderFactory")
    def test_indexing_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
    ) -> None:
        """Test the indexing pipeline component interactions.

        Validates the WeaviateMmrIndexingPipeline initialization and execution
        with mocked dependencies. Ensures proper document loading, embedding
        generation, and database upsert operations.

        Args:
            mock_dataloader_catalog: Mocked DataloaderCatalog for document retrieval
            mock_embedder_factory: Mocked EmbedderFactory for embedder creation
            mock_db_cls: Mocked WeaviateVectorDB class for database operations

        Returns:
            None

        Raises:
            AssertionError: If document counts or mock call assertions fail

        Mock Configuration:
            - Documents: 2 test documents with metadata
            - Embeddings: 384-dimensional vectors (MiniLM-compatible)
            - Database: Successful upsert returning document count

        Expected Behavior:
            - Pipeline loads documents via DataloaderCatalog
            - Documents are embedded using the configured embedder
            - Embedded documents are upserted to WeaviateVectorDB
            - Returns accurate count of indexed documents
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
            "weaviate": {
                "url": "http://localhost:8080",
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

        pipeline = WeaviateMmrIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        mock_dataloader_catalog.create.assert_called_once()
        mock_embedder.run.assert_called_once()
        mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.mmr.search.weaviate.RAGHelper")
    @patch("vectordb.haystack.mmr.search.weaviate.RerankerFactory")
    @patch("vectordb.haystack.mmr.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.mmr.search.weaviate.EmbedderFactory")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_ranker_factory: MagicMock,
        mock_rag_helper: MagicMock,
    ) -> None:
        """Test the search pipeline with MMR reranking.

        Validates the WeaviateMmrSearchPipeline search operation including
        query embedding, candidate retrieval, MMR-based diversity reranking,
        and optional RAG generation.

        Args:
            mock_embedder_factory: Mocked EmbedderFactory for text embedder
            mock_db_cls: Mocked WeaviateVectorDB class for candidate retrieval
            mock_ranker_factory: Mocked RerankerFactory for diversity ranker
            mock_rag_helper: Mocked RAGHelper for generative answering

        Returns:
            None

        Raises:
            AssertionError: If document counts or RAG generation assertions fail

        Mock Configuration:
            - Query embedding: 384-dimensional vector
            - Candidates: 10 documents with incremental embeddings
            - Reranking: Top 5 diverse documents selected
            - RAG: Mocked generator producing "Generated answer"

        Expected Behavior:
            - Query is embedded using the configured text embedder
            - WeaviateVectorDB returns candidate documents via similarity search
            - Diversity ranker applies MMR algorithm to select diverse top-k
            - RAG generator produces answer using diverse context
            - Returns both reranked documents and generated answer
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
            "weaviate": {
                "url": "http://localhost:8080",
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

        pipeline = WeaviateMmrSearchPipeline(config)
        result = pipeline.search("What is AI?", top_k=5, top_k_candidates=20)

        assert len(result["documents"]) == 5
        assert result["answer"] == "Generated answer"
        mock_ranker.run.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(not os.getenv("WEAVIATE_URL"), reason="WEAVIATE_URL not set")
    def test_indexing_integration(self) -> None:
        """Integration test for end-to-end document indexing.

        Executes the full WeaviateMmrIndexingPipeline against a live Weaviate
        instance. Tests document loading, embedding generation, and database
        operations in a real environment.

        Environment Requirements:
            - WEAVIATE_URL: Valid URL to running Weaviate instance
            - Network access to Weaviate gRPC/HTTP endpoints

        Configuration:
            - Collection: "mmr_integration_test" (created with recreate=True)
            - Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
            - Data source: TriviaQA dataloader with 5-document limit

        Returns:
            None

        Raises:
            AssertionError: If no documents are indexed
            ConnectionError: If WEAVIATE_URL is unreachable (pytest handles as failure)

        Expected Behavior:
            - Connects to live Weaviate instance at WEAVIATE_URL
            - Loads documents from TriviaQA dataset
            - Generates embeddings using MiniLM model
            - Creates or recreates collection with specified dimension
            - Upserts documents and returns positive document count

        Cleanup:
            Collection "mmr_integration_test" is recreated each run,
            preventing accumulation of test data.
        """
        config: dict[str, Any] = {
            "weaviate": {
                "url": os.getenv("WEAVIATE_URL"),
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

        pipeline = WeaviateMmrIndexingPipeline(config)
        result = pipeline.run()
        assert result["documents_indexed"] > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("WEAVIATE_URL") or not os.getenv("GROQ_API_KEY"),
        reason="WEAVIATE_URL or GROQ_API_KEY not set",
    )
    def test_search_with_rag_integration(self) -> None:
        """Integration test for MMR search with RAG generation.

        Executes the full WeaviateMmrSearchPipeline including MMR-based
        diversity reranking and RAG generation against live services.
        Validates end-to-end retrieval-augmented generation workflow.

        Environment Requirements:
            - WEAVIATE_URL: Valid URL to running Weaviate instance
            - GROQ_API_KEY: Valid API key for Groq LLM service
            - Pre-indexed data: Run test_indexing_integration first

        Configuration:
            - Collection: "mmr_integration_test" (shared with indexing test)
            - MMR lambda: 0.5 (balance between relevance and diversity)
            - LLM: llama-3.3-70b-versatile via Groq API
            - Query: "What is machine learning?"

        Returns:
            None

        Raises:
            AssertionError: If search returns no documents
            ConnectionError: If Weaviate or Groq services are unreachable
            KeyError: If response structure is invalid

        Expected Behavior:
            - Connects to Weaviate and queries indexed documents
            - Retrieves top-k candidates using similarity search
            - Applies MMR diversity reranking with lambda_threshold=0.5
            - Sends diverse context to Groq LLM via RAG pipeline
            - Returns both reranked documents and generated answer

        Prerequisites:
            Requires documents to be indexed in "mmr_integration_test"
            collection. Run test_indexing_integration first to populate data.

        Cost Considerations:
            This test makes live LLM API calls to Groq, incurring token costs.
            The test is marked as integration and skipped by default.
        """
        config: dict[str, Any] = {
            "weaviate": {
                "url": os.getenv("WEAVIATE_URL"),
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

        pipeline = WeaviateMmrSearchPipeline(config)
        result = pipeline.search("What is machine learning?", top_k=5)
        assert "documents" in result
        assert result["documents"] is not None
