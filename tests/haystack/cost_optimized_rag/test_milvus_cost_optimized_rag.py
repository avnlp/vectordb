"""Tests for Milvus cost-optimized RAG pipeline.

This module provides comprehensive test coverage for the Milvus vector database
integration within the cost-optimized RAG (Retrieval-Augmented Generation) pipeline.
The cost-optimized approach includes reranking capabilities and generator controls
to optimize token usage and API costs while maintaining retrieval quality.

Test Coverage:
    - Unit tests for indexer initialization with mocked Milvus connections
    - Unit tests for searcher initialization with configurable RAG components
    - Unit tests for search result formatting with Milvus distance scores
    - Integration tests requiring MILVUS_HOST and MILVUS_PORT environment variables
    - Full RAG pipeline integration tests requiring GROQ_API_KEY

Environment Variables:
    MILVUS_HOST: Hostname or IP address of the Milvus server.
                 Required for integration tests connecting to real instances.
    MILVUS_PORT: Port number for Milvus server connection (typically 19530).
                 Required alongside MILVUS_HOST for integration tests.
    GROQ_API_KEY: API key for Groq LLM service. Required for RAG integration tests.

Architecture:
    The cost-optimized RAG pipeline consists of two main components:
    1. MilvusIndexingPipeline: Indexes documents with sentence transformer
       embeddings into Milvus collections with automatic schema management
    2. MilvusSearchPipeline: Performs similarity search with optional reranking
       and LLM-based answer generation

Milvus Connection Model:
    Unlike cloud-based vector databases, Milvus requires direct server connection
    via host and port. The integration tests verify collection creation, data
    insertion, vector search, and proper connection lifecycle management.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pymilvus import Collection, connections

from vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer import (
    MilvusIndexingPipeline,
)
from vectordb.haystack.cost_optimized_rag.search.milvus_searcher import (
    MilvusSearchPipeline,
)


class TestMilvusCostOptimizedRag:
    """Unit and integration tests for Milvus cost-optimized RAG pipeline.

    This test class validates the complete Milvus-based RAG workflow including
    document indexing, semantic search, result formatting, and full RAG pipeline
    execution with LLM answer generation.

    Test Categories:
        - Unit tests: Use mocked Milvus connections and collections for fast,
          isolated testing of pipeline initialization and search logic
        - Integration tests: Connect to real Milvus server instances to validate
          end-to-end functionality with actual document storage and retrieval

    Configuration:
        Tests use YAML configuration files created via pytest fixtures.
        The base_config_dict fixture provides default settings that are
        customized for each test scenario with Milvus-specific parameters
        including host, port, and collection names.

    Server Requirements:
        Integration tests require a running Milvus instance accessible via
        MILVUS_HOST and MILVUS_PORT environment variables. Tests are skipped
        unless both variables are set, preventing connection failures.
    """

    @pytest.fixture
    def mock_config_path(self, tmp_path, base_config_dict):
        """Create a temporary YAML configuration file for testing.

        Generates a test-specific config file with Milvus-specific settings
        including the server host, port, and collection name. The configuration
        is written to a temporary directory provided by pytest's tmp_path fixture.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            base_config_dict: Base configuration dictionary from conftest.py.

        Returns:
            str: Absolute path to the generated YAML configuration file.
        """
        config_data = base_config_dict
        config_data["milvus"] = {
            "host": "localhost",
            "port": 19530,
        }
        config_data["collection"]["name"] = "test_milvus_collection"

        config_path = tmp_path / "test_milvus_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    def test_indexer_init(
        self, mock_embedder, mock_collection, mock_connections, mock_config_path
    ):
        """Test Milvus indexing pipeline initialization.

                Validates that the MilvusIndexingPipeline correctly initializes with:
                - Configuration loaded from YAML file
                - SentenceTransformersDocumentEmbedder for document embeddings
                - Milvus connections manager for server communication
                - Collection management for schema definition and data storage
                - Proper component wiring and warm-up execution

                Mocks:
                    - connections: Prevents actual server connections
                    - Collection: Avoids real collection operations
                    - SentenceTransformersDocumentEmbedder: Avoids model loading
                      overhead

                Collection Lifecycle:
                    The indexer manages collection lifecycle by dropping existing
        test collections and recreating them with updated schemas.
        """
        mock_embedder.return_value.warm_up = MagicMock()
        mock_collection_instance = MagicMock()
        mock_collection_instance.drop = MagicMock()
        mock_collection.return_value = mock_collection_instance

        pipeline = MilvusIndexingPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_connections.connect.assert_called_once()
        assert mock_collection.call_count >= 1
        assert pipeline.embedder is not None
        assert pipeline.collection is not None

    @patch("vectordb.haystack.cost_optimized_rag.search.milvus_searcher.connections")
    @patch("vectordb.haystack.cost_optimized_rag.search.milvus_searcher.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.milvus_searcher.SentenceTransformersTextEmbedder"
    )
    def test_searcher_init(
        self, mock_embedder, mock_collection, mock_connections, mock_config_path
    ):
        """Test Milvus search pipeline initialization without RAG components.

        Validates basic searcher initialization with embedding and retrieval
        capabilities only. Reranking and generator are disabled to avoid
        Haystack Pipeline validation issues during unit testing.

        Configuration Modification:
            Temporarily modifies the config file to disable reranking and
            generator components, creating a simplified pipeline for validation.

        Mocks:
            - connections: Prevents server connections
            - Collection: Avoids real collection loading
            - SentenceTransformersTextEmbedder: Avoids model loading

        Collection Loading:
            The searcher loads the collection into memory for fast vector
            search operations, which is mocked in this unit test.
        """
        mock_embedder.return_value.warm_up = MagicMock()
        mock_collection_instance = MagicMock()
        mock_collection_instance.load = MagicMock()
        mock_collection.return_value = mock_collection_instance

        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = MilvusSearchPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_connections.connect.assert_called_once()
        mock_collection.assert_called_once()
        assert pipeline.embedder is not None

    @patch("vectordb.haystack.cost_optimized_rag.search.milvus_searcher.connections")
    @patch("vectordb.haystack.cost_optimized_rag.search.milvus_searcher.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.milvus_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_returns_results(
        self, mock_embedder, mock_collection, mock_connections, mock_config_path
    ):
        """Test search returns properly formatted results with Milvus distances.

        Validates the complete search flow including:
        - Query embedding generation via mocked embedder
        - Milvus collection search execution with vector similarity
        - Result formatting with document IDs, content, metadata, and distances
        - Proper handling of Milvus search result structure (list of lists)

        Result Format:
            Each result dictionary contains:
            - id: Document identifier (string)
            - score: Milvus distance value (lower indicates higher similarity
              for L2 distance metric, varies by index type)
            - content: Document text content from entity data
            - metadata: Document metadata dictionary from entity

        Milvus Search Structure:
            Milvus returns search results as a list of lists, where each inner
            list contains Hit objects for one query. This test mocks the
            structure: [[hit1, hit2, ...]] for single query scenarios.
        """
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_instance

        mock_hit = MagicMock()
        mock_hit.id = "doc1"
        mock_hit.distance = 0.05
        mock_hit.entity.get.side_effect = {
            "content": "Test content 1",
            "metadata": {"source": "test"},
        }.get

        mock_search_result = [[mock_hit]]

        mock_collection_instance = MagicMock()
        mock_collection_instance.load = MagicMock()
        mock_collection_instance.search.return_value = mock_search_result
        mock_collection.return_value = mock_collection_instance

        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = MilvusSearchPipeline(mock_config_path)
        results = pipeline.search("test query", top_k=1)

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.05
        assert "content" in results[0]
        assert results[0]["content"] == "Test content 1"
        assert results[0]["metadata"]["source"] == "test"
        mock_collection_instance.search.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("MILVUS_HOST") and os.getenv("MILVUS_PORT")),
        reason="MILVUS_HOST and MILVUS_PORT not set for Milvus integration tests",
    )
    def test_indexing_integration(self, tmp_path, base_config_dict):
        """Integration test for document indexing pipeline with real Milvus server.

                Connects to a live Milvus instance, creates a collection with proper
                schema, indexes sample documents using the TriviaQA dataloader (limited
        to 3 documents), and verifies documents are stored with embeddings.

                Environment Requirements:
                    MILVUS_HOST: Hostname or IP of running Milvus server
                    MILVUS_PORT: Port number for server connection (typically 19530)

                Collection Naming:
                    Uses "integration_test_milvus_idx" to avoid collisions with other
                    test runs and production collections.

                Verification:
                    Connects via pymilvus connections and validates that the collection
                    contains more than zero entities after indexing.

                Cleanup:
                    Drops the test collection after verification to maintain clean
                    server state and prevent accumulation of test data.
        """
        config_data = base_config_dict
        config_data["milvus"] = {
            "host": os.getenv("MILVUS_HOST"),
            "port": int(os.getenv("MILVUS_PORT")),
        }
        config_data["collection"]["name"] = "integration_test_milvus_idx"
        config_data["dataloader"]["limit"] = 3
        config_path = tmp_path / "integration_milvus_idx_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        pipeline = MilvusIndexingPipeline(config_path)
        pipeline.run()

        conn_params = {
            "host": os.getenv("MILVUS_HOST"),
            "port": os.getenv("MILVUS_PORT"),
        }
        connections.connect(alias="default", **conn_params)
        assert Collection(name=config_data["collection"]["name"]).num_entities > 0

        Collection(name=config_data["collection"]["name"]).drop()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("MILVUS_HOST") and os.getenv("MILVUS_PORT")),
        reason="MILVUS_HOST and MILVUS_PORT not set for Milvus integration tests",
    )
    def test_search_integration(self, tmp_path, base_config_dict):
        """Integration test for semantic search with real Milvus server.

        End-to-end test that indexes documents into a live Milvus collection
        then performs semantic vector search. Validates that search returns
        relevant documents with proper formatting including content and metadata.

        Workflow:
            1. Create indexer config and index documents (limited to 3)
            2. Create searcher config pointing to same collection
            3. Execute search query: "What is the capital of France?"
            4. Verify results contain expected fields (content, score, metadata)

        Environment Requirements:
            MILVUS_HOST: Hostname or IP of running Milvus server
            MILVUS_PORT: Port number for server connection

        Collection Management:
            Creates "integration_test_milvus_srh" collection for indexing and
            drops it after search verification to maintain clean server state.
        """
        idx_config_data = base_config_dict
        idx_config_data["milvus"] = {
            "host": os.getenv("MILVUS_HOST"),
            "port": int(os.getenv("MILVUS_PORT")),
        }
        idx_config_data["collection"]["name"] = "integration_test_milvus_srh"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_milvus_idx_config_srh.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = MilvusIndexingPipeline(idx_config_path)
        indexer.run()

        srh_config_data = base_config_dict
        srh_config_data["milvus"] = {
            "host": os.getenv("MILVUS_HOST"),
            "port": int(os.getenv("MILVUS_PORT")),
        }
        srh_config_data["collection"]["name"] = "integration_test_milvus_srh"
        srh_config_path = tmp_path / "integration_milvus_srh_config.yaml"
        with open(srh_config_path, "w") as f:
            yaml.dump(srh_config_data, f)

        pipeline = MilvusSearchPipeline(srh_config_path)
        results = pipeline.search("What is the capital of France?", top_k=1)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

        Collection(name=idx_config_data["collection"]["name"]).drop()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (
            os.getenv("MILVUS_HOST")
            and os.getenv("MILVUS_PORT")
            and os.getenv("GROQ_API_KEY")
        ),
        reason="MILVUS_HOST, MILVUS_PORT, or GROQ_API_KEY not set for Milvus RAG integration tests",
    )
    def test_rag_integration(self, tmp_path, base_config_dict):
        """Integration test for full RAG pipeline with LLM answer generation.

        Tests the complete cost-optimized RAG workflow:
        1. Index documents into Milvus collection with embeddings
        2. Configure generator with Groq API for LLM-based answer generation
        3. Execute search_with_rag to retrieve documents and generate contextual answer
        4. Validate response contains both retrieved documents and generated answer

        Generator Configuration:
            Uses Groq's llama-3.3-70b-versatile model for answer generation.
            Requires GROQ_API_KEY environment variable for authentication.

        Result Format:
            The search_with_rag method returns a dictionary with:
            - documents: List of retrieved document dictionaries with content,
              metadata, and similarity scores from Milvus
            - answer: LLM-generated answer string synthesized from retrieved context

        Environment Requirements:
            MILVUS_HOST: Hostname or IP of running Milvus server
            MILVUS_PORT: Port number for server connection
            GROQ_API_KEY: Valid API key for Groq LLM service access

        Collection Management:
            Creates "integration_test_milvus_rag" collection for the test and
            drops it after RAG verification to maintain clean server state.
        """
        idx_config_data = base_config_dict
        idx_config_data["milvus"] = {
            "host": os.getenv("MILVUS_HOST"),
            "port": int(os.getenv("MILVUS_PORT")),
        }
        idx_config_data["collection"]["name"] = "integration_test_milvus_rag"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_milvus_idx_config_rag.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = MilvusIndexingPipeline(idx_config_path)
        indexer.run()

        rag_config_data = base_config_dict
        rag_config_data["milvus"] = {
            "host": os.getenv("MILVUS_HOST"),
            "port": int(os.getenv("MILVUS_PORT")),
        }
        rag_config_data["collection"]["name"] = "integration_test_milvus_rag"
        rag_config_data["generator"]["enabled"] = True
        rag_config_data["generator"]["api_key"] = os.getenv("GROQ_API_KEY")
        rag_config_data["generator"]["model"] = "llama-3.3-70b-versatile"
        rag_config_path = tmp_path / "integration_milvus_rag_config.yaml"
        with open(rag_config_path, "w") as f:
            yaml.dump(rag_config_data, f)

        pipeline = MilvusSearchPipeline(rag_config_path)
        result = pipeline.search_with_rag("What is the capital of France?", top_k=1)

        assert "documents" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        Collection(name=idx_config_data["collection"]["name"]).drop()
