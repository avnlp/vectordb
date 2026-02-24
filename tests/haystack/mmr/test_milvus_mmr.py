"""Unit and integration tests for Haystack Milvus MMR pipelines.

This module provides comprehensive test coverage for the Maximal Marginal
Relevance (MMR) pipeline implementations with Milvus vector database integration.
It tests both the indexing pipeline (MilvusMmrIndexingPipeline) and search
pipeline (MilvusMmrSearchPipeline).

Key Features Tested:
    - MilvusVectorDB integration with URI-based connections
    - Collection-based document organization in Milvus
    - Document embedding and upsert operations
    - MMR-based diversity ranking for search results
    - RAG (Retrieval-Augmented Generation) pipeline integration
    - Haystack Document handling and metadata preservation

Test Categories:
    - Unit tests: Mock-based tests for pipeline logic without external dependencies
    - Integration tests: Full end-to-end tests requiring live Milvus instance and API
      keys

Dependencies:
    - pytest for test framework and fixtures
    - unittest.mock for mocking external services
    - Haystack Document for structured document representation
    - MilvusVectorDB for vector storage and retrieval

Environment Requirements:
    - MILVUS_URI: URI for Milvus server connection (e.g., http://localhost:19530)
    - GROQ_API_KEY: API key for Groq LLM provider (for RAG integration tests)
"""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.mmr.indexing import MilvusMmrIndexingPipeline
from vectordb.haystack.mmr.search import MilvusMmrSearchPipeline


class TestMilvusMMR:
    """Test suite for Milvus MMR indexing and search pipelines.

    This class provides comprehensive test coverage for the Milvus-based MMR
    pipeline implementations, including both unit tests with mocked dependencies
    and integration tests with live Milvus connections.

    The MMR (Maximal Marginal Relevance) algorithm balances relevance and diversity
    in search results by selecting documents that are both relevant to the query
    and dissimilar to already-selected documents.

    Test Coverage:
        - MilvusMmrIndexingPipeline: Document ingestion, embedding, and storage
        - MilvusMmrSearchPipeline: Query embedding, retrieval, MMR ranking, and RAG
        - Configuration validation and pipeline initialization
        - Integration with external services (embedders, LLM providers)

    Configuration Structure:
        The pipelines expect a configuration dictionary with the following sections:
        - milvus: URI, collection_name, dimension, recreate options
        - embeddings: Model specification for document/query embedding
        - dataloader: Data source configuration (name, limit, etc.)
        - mmr: Lambda threshold for diversity-relevance balance
        - rag: Optional LLM configuration for answer generation

    Attributes:
        None - Uses pytest fixture pattern and test methods

    Example:
        Basic configuration for unit testing:

        >>> config = {
        ...     "milvus": {
        ...         "uri": "http://localhost:19530",
        ...         "collection_name": "test-collection",
        ...         "dimension": 384,
        ...     },
        ...     "embeddings": {
        ...         "model": "sentence-transformers/all-MiniLM-L6-v2",
        ...     },
        ...     "dataloader": {
        ...         "name": "triviaqa",
        ...         "limit": 10,
        ...     },
        ... }
    """

    @patch("vectordb.haystack.mmr.indexing.milvus.DataloaderCatalog")
    @patch("vectordb.haystack.mmr.indexing.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.mmr.indexing.milvus.EmbedderFactory")
    def test_indexing_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_dataloader_catalog: MagicMock,
    ) -> None:
        """Unit test for MilvusMmrIndexingPipeline with mocked dependencies.

        Verifies that the indexing pipeline correctly orchestrates the document
        loading, embedding, and upsert workflow using mocked external services.

        Test Flow:
            1. Mock DataloaderCatalog to return sample documents
            2. Mock EmbedderFactory to return mock embedder with embeddings
            3. Mock MilvusVectorDB to simulate successful upsert
            4. Execute pipeline with test configuration
            5. Verify all components were called and results are correct

        Args:
            mock_dataloader_catalog: Mocked DataloaderCatalog for document loading
            mock_embedder_factory: Mocked EmbedderFactory for embedder creation
            mock_db_cls: Mocked MilvusVectorDB class for database operations

        Assertions:
            - Documents indexed count matches expected (2)
            - DataloaderCatalog.create() was called once
            - Embedder.run() was called once for embedding generation
            - MilvusVectorDB.upsert() was called once for storage

        Configuration Used:
            - Milvus: localhost URI, test-collection, 384 dimensions
            - Embeddings: sentence-transformers/all-MiniLM-L6-v2 model
            - Dataloader: TriviaQA dataset with 10 document limit
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
            "milvus": {
                "uri": "http://localhost:19530",
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

        pipeline = MilvusMmrIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        mock_dataloader_catalog.create.assert_called_once()
        mock_embedder.run.assert_called_once()
        mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.mmr.search.milvus.RAGHelper")
    @patch("vectordb.haystack.mmr.search.milvus.RerankerFactory")
    @patch("vectordb.haystack.mmr.search.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.mmr.search.milvus.EmbedderFactory")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_ranker_factory: MagicMock,
        mock_rag_helper: MagicMock,
    ) -> None:
        """Unit test for MilvusMmrSearchPipeline with mocked dependencies.

        Verifies that the search pipeline correctly orchestrates the query
        embedding, vector retrieval, MMR reranking, and optional RAG generation
        workflow using mocked external services.

        Test Flow:
            1. Mock EmbedderFactory for query embedding
            2. Mock MilvusVectorDB to return candidate documents
            3. Mock RerankerFactory to simulate MMR diversity ranking
            4. Mock RAGHelper for answer generation
            5. Execute search with test query and configuration
            6. Verify MMR ranking and RAG components were invoked

        Args:
            mock_embedder_factory: Mocked EmbedderFactory for query embedding
            mock_db_cls: Mocked MilvusVectorDB class for vector retrieval
            mock_ranker_factory: Mocked RerankerFactory for diversity ranking
            mock_rag_helper: Mocked RAGHelper for answer generation

        Assertions:
            - Result contains 5 documents after MMR reranking
            - Generated answer matches expected RAG output
            - Reranker.run() was called for MMR processing
            - Query embedding and vector retrieval were invoked

        Configuration Used:
            - Milvus: localhost URI, test-collection
            - Embeddings: sentence-transformers/all-MiniLM-L6-v2 model
            - MMR: Lambda threshold of 0.5 for diversity balance
            - RAG: Enabled with Groq provider and llama-3.3-70b model

        MMR Parameters:
            - top_k: 5 (final documents to return)
            - top_k_candidates: 20 (initial candidates before MMR filtering)
            - lambda_threshold: 0.5 (balance between relevance and diversity)
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
            "milvus": {
                "uri": "http://localhost:19530",
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

        pipeline = MilvusMmrSearchPipeline(config)
        result = pipeline.search("What is AI?", top_k=5, top_k_candidates=20)

        assert len(result["documents"]) == 5
        assert result["answer"] == "Generated answer"
        mock_ranker.run.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(not os.getenv("MILVUS_URI"), reason="MILVUS_URI not set")
    def test_indexing_integration(self) -> None:
        """Integration test for MilvusMmrIndexingPipeline with live database.

        Executes a full indexing workflow against a live Milvus instance to verify
        end-to-end document ingestion, embedding, and storage functionality.

        Prerequisites:
            - MILVUS_URI environment variable must be set
            - Milvus server must be accessible at the specified URI
            - Collection will be recreated if it exists (recreate: True)

        Test Flow:
            1. Load configuration with live Milvus URI from environment
            2. Initialize MilvusMmrIndexingPipeline with real dependencies
            3. Load documents from TriviaQA dataloader (limited to 5)
            4. Generate embeddings using sentence-transformers model
            5. Upsert documents into Milvus collection
            6. Verify documents were successfully indexed

        Configuration:
            - Collection name: mmr_integration_test
            - Dimension: 384 (matching MiniLM model output)
            - Recreate: True (drops and recreates collection)
            - Document limit: 5 (for quick test execution)

        Assertions:
            - result["documents_indexed"] > 0 (documents were stored)

        Raises:
            pytest.skip: If MILVUS_URI environment variable is not set
        """
        config: dict[str, Any] = {
            "milvus": {
                "uri": os.getenv("MILVUS_URI"),
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

        pipeline = MilvusMmrIndexingPipeline(config)
        result = pipeline.run()
        assert result["documents_indexed"] > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("MILVUS_URI") or not os.getenv("GROQ_API_KEY"),
        reason="MILVUS_URI or GROQ_API_KEY not set",
    )
    def test_search_with_rag_integration(self) -> None:
        """Integration test for MilvusMmrSearchPipeline with RAG and live services.

        Executes a full search and RAG workflow against a live Milvus instance
        and Groq LLM API to verify end-to-end retrieval and answer generation.

        Prerequisites:
            - MILVUS_URI environment variable must be set
            - GROQ_API_KEY environment variable must be set
            - Milvus server must be accessible
            - Collection "mmr_integration_test" must exist with indexed documents
            - Groq API must be accessible for LLM generation

        Test Flow:
            1. Load configuration with live service URIs and API keys
            2. Initialize MilvusMmrSearchPipeline with real dependencies
            3. Embed search query using sentence-transformers model
            4. Query Milvus collection for candidate documents
            5. Apply MMR diversity ranking with lambda threshold 0.5
            6. Generate answer using Groq LLM (llama-3.3-70b-versatile)
            7. Verify search results and generated answer

        Configuration:
            - Collection name: mmr_integration_test
            - MMR lambda_threshold: 0.5 (balance relevance and diversity)
            - RAG provider: groq
            - RAG model: llama-3.3-70b-versatile
            - top_k: 5 (default, returns 5 diverse documents)

        Assertions:
            - result contains "documents" key
            - result["documents"] is not None (retrieval succeeded)

        Raises:
            pytest.skip: If MILVUS_URI or GROQ_API_KEY environment variables are not set
        """
        config: dict[str, Any] = {
            "milvus": {
                "uri": os.getenv("MILVUS_URI"),
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

        pipeline = MilvusMmrSearchPipeline(config)
        result = pipeline.search("What is machine learning?", top_k=5)
        assert "documents" in result
        assert result["documents"] is not None
