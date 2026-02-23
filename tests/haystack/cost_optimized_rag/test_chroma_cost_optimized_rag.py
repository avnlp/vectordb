"""Tests for Chroma cost-optimized RAG pipeline.

This module provides comprehensive test coverage for the Chroma vector database
integration within the cost-optimized RAG (Retrieval-Augmented Generation) pipeline.
The cost-optimized approach includes reranking capabilities and generator controls
to optimize token usage and API costs while maintaining retrieval quality.

Test Coverage:
    - Unit tests for indexer initialization with mocked ChromaDB client
    - Unit tests for searcher initialization with configurable RAG components
    - Unit tests for search result formatting and score conversion
    - Integration tests requiring CHROMADB_PATH environment variable
    - Full RAG pipeline integration tests requiring GROQ_API_KEY

Environment Variables:
    CHROMADB_PATH: Path to temporary directory for Chroma persistent storage.
                   Required for integration tests to ensure safe file operations.
    GROQ_API_KEY: API key for Groq LLM service. Required for RAG integration tests.

Architecture:
    The cost-optimized RAG pipeline consists of two main components:
    1. ChromaIndexingPipeline: Indexes documents with sentence transformer embeddings
    2. ChromaSearchPipeline: Performs similarity search with optional reranking
       and LLM-based answer generation

Note:
    Chroma integration tests use PersistentClient which creates files on disk.
    These tests are skipped by default unless CHROMADB_PATH is explicitly set
to a temporary directory for CI safety.
"""

import os
from unittest.mock import MagicMock, patch

import chromadb
import pytest
import yaml

from vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer import (
    ChromaIndexingPipeline,
)
from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
    ChromaSearchPipeline,
)


class TestChromaCostOptimizedRag:
    """Unit and integration tests for Chroma cost-optimized RAG pipeline.

    This test class validates the complete Chroma-based RAG workflow including
    document indexing, semantic search, result formatting, and full RAG pipeline
    execution with LLM answer generation.

    Test Categories:
        - Unit tests: Use mocked ChromaDB client and embedders for fast,
          isolated testing of pipeline initialization and search logic
        - Integration tests: Connect to real ChromaDB instances to validate
          end-to-end functionality with actual document storage and retrieval

    Configuration:
        Tests use YAML configuration files created via pytest fixtures.
        The base_config_dict fixture provides default settings that are
        customized for each test scenario.
    """

    @pytest.fixture
    def mock_config_path(self, tmp_path, base_config_dict):
        """Create a temporary YAML configuration file for testing.

        Generates a test-specific config file with Chroma-specific settings
        including the database path and collection name. The configuration
        is written to a temporary directory provided by pytest's tmp_path.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            base_config_dict: Base configuration dictionary from conftest.py.

        Returns:
            str: Absolute path to the generated YAML configuration file.
        """
        config_data = base_config_dict
        config_data["chroma"] = {
            "path": str(tmp_path / "chroma_db"),
        }
        config_data["collection"]["name"] = "test_chroma_collection"

        config_path = tmp_path / "test_chroma_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.chromadb.PersistentClient"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    def test_indexer_init(self, mock_embedder, mock_chroma_client, mock_config_path):
        """Test Chroma indexing pipeline initialization.

        Validates that the ChromaIndexingPipeline correctly initializes with:
        - Configuration loaded from YAML file
        - SentenceTransformersDocumentEmbedder for document embeddings
        - ChromaDB PersistentClient for vector storage
        - Proper component wiring and warm-up execution

        Mocks:
            - chromadb.PersistentClient: Prevents actual database connections
            - SentenceTransformersDocumentEmbedder: Avoids model loading overhead
        """
        mock_embedder.return_value.warm_up = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get_or_create_collection.return_value = MagicMock()
        mock_chroma_client.return_value = mock_client_instance

        pipeline = ChromaIndexingPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_chroma_client.assert_called_once()
        assert pipeline.embedder is not None
        assert pipeline.client is not None

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    def test_searcher_init(self, mock_embedder, mock_chroma_client, mock_config_path):
        """Test Chroma search pipeline initialization without RAG components.

        Validates basic searcher initialization with embedding and retrieval
        capabilities disabled. Reranking and generator are disabled to avoid
        Haystack Pipeline validation issues during unit testing.

        Configuration Modification:
            Temporarily modifies the config file to disable reranking and
            generator components for simplified pipeline validation.

        Mocks:
            - chromadb.PersistentClient: Prevents database connections
            - SentenceTransformersTextEmbedder: Avoids model loading
        """
        mock_embedder.return_value.warm_up = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = MagicMock()
        mock_chroma_client.return_value = mock_client_instance

        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = ChromaSearchPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_chroma_client.assert_called_once()
        assert pipeline.embedder is not None

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.chromadb.PersistentClient"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.chroma_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_returns_results(
        self, mock_embedder, mock_chroma_client, mock_config_path
    ):
        """Test search returns properly formatted results with score conversion.

        Validates the complete search flow including:
        - Query embedding generation via mocked embedder
        - ChromaDB collection query execution
        - Result formatting with document IDs, content, and metadata
        - Distance-to-score conversion (1.0 / (1.0 + distance))

        Score Conversion:
            Chroma returns distances (lower is better), but the pipeline
            converts these to scores (higher is better) using:
            score = 1.0 / (1.0 + distance)

        Result Format:
            Each result dictionary contains:
            - id: Document identifier
            - score: Converted similarity score (0.0 to 1.0)
            - content: Document text content
            - metadata: Document metadata dictionary
        """
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_instance

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "embeddings": None,
            "documents": [["Test content 1", "Test content 2"]],
            "metadatas": [[{"source": "test"}, {"source": "test"}]],
            "distances": [[0.05, 0.1]],
        }
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client_instance

        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = ChromaSearchPipeline(mock_config_path)
        results = pipeline.search("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert abs(results[0]["score"] - (1.0 / (1.0 + 0.05))) < 1e-6
        assert "content" in results[0]
        assert results[0]["content"] == "Test content 1"
        assert results[0]["metadata"]["source"] == "test"
        mock_collection.query.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("CHROMADB_PATH"),
        reason="CHROMADB_PATH not set for Chroma integration tests",
    )
    def test_indexing_integration(self, tmp_path, base_config_dict):
        """Integration test for document indexing pipeline with real ChromaDB.

        Creates a real ChromaDB collection, indexes sample documents using the
        TriviaQA dataloader (limited to 3 documents), and verifies the
        documents are stored correctly with embeddings.

        Environment Requirements:
            CHROMADB_PATH must be set to a writable temporary directory.

        Cleanup:
            Deletes the test collection after verification to prevent
            accumulation of test data.
        """
        chroma_db_path = tmp_path / "chroma_db_idx"
        chroma_db_path.mkdir(exist_ok=True)

        config_data = base_config_dict
        config_data["chroma"] = {
            "path": str(chroma_db_path),
        }
        config_data["collection"]["name"] = "integration_test_chroma_idx"
        config_data["dataloader"]["limit"] = 3
        config_path = tmp_path / "integration_chroma_idx_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        pipeline = ChromaIndexingPipeline(config_path)
        pipeline.run()

        client = chromadb.PersistentClient(path=str(chroma_db_path))
        collection = client.get_collection(name=config_data["collection"]["name"])
        assert collection.count() > 0

        client.delete_collection(config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("CHROMADB_PATH"),
        reason="CHROMADB_PATH not set for Chroma integration tests",
    )
    def test_search_integration(self, tmp_path, base_config_dict):
        """Integration test for semantic search with real ChromaDB.

        End-to-end test that indexes documents then performs semantic search.
        Validates that search returns relevant documents with proper formatting.

        Workflow:
            1. Create indexer config and index documents (limited to 3)
            2. Create searcher config pointing to same collection
            3. Execute search query: "What is the capital of France?"
            4. Verify results contain expected fields (content, score, metadata)

        Environment Requirements:
            CHROMADB_PATH must be set to a writable temporary directory.
        """
        chroma_db_path = tmp_path / "chroma_db_srh"
        chroma_db_path.mkdir(exist_ok=True)

        idx_config_data = base_config_dict
        idx_config_data["chroma"] = {
            "path": str(chroma_db_path),
        }
        idx_config_data["collection"]["name"] = "integration_test_chroma_srh"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_chroma_idx_config_srh.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = ChromaIndexingPipeline(idx_config_path)
        indexer.run()

        srh_config_data = base_config_dict
        srh_config_data["chroma"] = {
            "path": str(chroma_db_path),
        }
        srh_config_data["collection"]["name"] = "integration_test_chroma_srh"
        srh_config_path = tmp_path / "integration_chroma_srh_config.yaml"
        with open(srh_config_path, "w") as f:
            yaml.dump(srh_config_data, f)

        pipeline = ChromaSearchPipeline(srh_config_path)
        results = pipeline.search("What is the capital of France?", top_k=1)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

        client = chromadb.PersistentClient(path=str(chroma_db_path))
        client.delete_collection(idx_config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("CHROMADB_PATH") and os.getenv("GROQ_API_KEY")),
        reason="CHROMADB_PATH or GROQ_API_KEY not set for Chroma RAG integration tests",
    )
    def test_rag_integration(self, tmp_path, base_config_dict):
        """Integration test for full RAG pipeline with LLM answer generation.

        Tests the complete cost-optimized RAG workflow:
        1. Index documents into ChromaDB
        2. Configure generator with Groq API for LLM-based answers
        3. Execute search_with_rag to retrieve documents and generate answer
        4. Validate response contains both retrieved documents and generated answer

        Generator Configuration:
            Uses Groq's llama-3.3-70b-versatile model for answer generation.
            Requires GROQ_API_KEY environment variable for authentication.

        Result Format:
            The search_with_rag method returns a dictionary with:
            - documents: List of retrieved document dictionaries
            - answer: LLM-generated answer string based on retrieved context

        Environment Requirements:
            CHROMADB_PATH: Writable temporary directory for Chroma storage
            GROQ_API_KEY: Valid API key for Groq LLM service access
        """
        chroma_db_path = tmp_path / "chroma_db_rag"
        chroma_db_path.mkdir(exist_ok=True)

        idx_config_data = base_config_dict
        idx_config_data["chroma"] = {
            "path": str(chroma_db_path),
        }
        idx_config_data["collection"]["name"] = "integration_test_chroma_rag"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_chroma_idx_config_rag.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = ChromaIndexingPipeline(idx_config_path)
        indexer.run()

        rag_config_data = base_config_dict
        rag_config_data["chroma"] = {
            "path": str(chroma_db_path),
        }
        rag_config_data["collection"]["name"] = "integration_test_chroma_rag"
        rag_config_data["generator"]["enabled"] = True
        rag_config_data["generator"]["api_key"] = os.getenv("GROQ_API_KEY")
        rag_config_data["generator"]["model"] = "llama-3.3-70b-versatile"
        rag_config_path = tmp_path / "integration_chroma_rag_config.yaml"
        with open(rag_config_path, "w") as f:
            yaml.dump(rag_config_data, f)

        pipeline = ChromaSearchPipeline(rag_config_path)
        result = pipeline.search_with_rag("What is the capital of France?", top_k=1)

        assert "documents" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        client = chromadb.PersistentClient(path=str(chroma_db_path))
        client.delete_collection(idx_config_data["collection"]["name"])
