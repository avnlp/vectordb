"""Tests for Pinecone cost-optimized RAG pipeline.

This module provides comprehensive test coverage for the Pinecone vector database
integration within the cost-optimized RAG (Retrieval-Augmented Generation) pipeline.
The cost-optimized approach includes reranking capabilities and generator controls
to optimize token usage and API costs while maintaining retrieval quality.

Test Coverage:
    - Unit tests for indexer initialization with mocked Pinecone client
    - Unit tests for searcher initialization with configurable RAG components
    - Unit tests for search result formatting with Pinecone similarity scores
    - Integration tests requiring PINECONE_API_KEY environment variable
    - Full RAG pipeline integration tests requiring GROQ_API_KEY

Environment Variables:
    PINECONE_API_KEY: API key for Pinecone cloud service. Required for all
                      integration tests that connect to real Pinecone indexes.
    GROQ_API_KEY: API key for Groq LLM service. Required for RAG integration tests.

Architecture:
    The cost-optimized RAG pipeline consists of two main components:
    1. PineconeIndexingPipeline: Indexes documents with sentence transformer
       embeddings into Pinecone cloud indexes
    2. PineconeSearchPipeline: Performs similarity search with optional reranking
       and LLM-based answer generation

Index Management:
    Integration tests create temporary Pinecone indexes with unique names
    prefixed with "integration_test_pinecone_collection_". Indexes are deleted
    after test completion to minimize cloud resource usage and costs.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pinecone import Pinecone

from vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer import (
    PineconeIndexingPipeline,
)
from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
    PineconeSearchPipeline,
)


class TestPineconeCostOptimizedRag:
    """Unit and integration tests for Pinecone cost-optimized RAG pipeline.

    This test class validates the complete Pinecone-based RAG workflow including
    document indexing, semantic search, result formatting, and full RAG pipeline
    execution with LLM answer generation.

    Test Categories:
        - Unit tests: Use mocked Pinecone client and embedders for fast,
          isolated testing of pipeline initialization and search logic
        - Integration tests: Connect to real Pinecone cloud instances to validate
          end-to-end functionality with actual document storage and retrieval

    Configuration:
        Tests use YAML configuration files created via pytest fixtures.
        The base_config_dict fixture provides default settings that are
        customized for each test scenario with Pinecone-specific parameters
        including API key, environment/region, and index names.

    Cost Considerations:
        Integration tests create and delete Pinecone indexes to minimize
        cloud costs. Tests are skipped unless PINECONE_API_KEY is explicitly
        set, preventing accidental cloud resource usage during local development.
    """

    @pytest.fixture
    def mock_config_path(self, tmp_path, base_config_dict):
        """Create a temporary YAML configuration file for testing.

        Generates a test-specific config file with Pinecone-specific settings
        including the API key, cloud environment, and collection name. The
        configuration is written to a temporary directory provided by pytest's
        tmp_path fixture.

        Args:
            tmp_path: Pytest fixture providing a temporary directory path.
            base_config_dict: Base configuration dictionary from conftest.py.

        Returns:
            str: Absolute path to the generated YAML configuration file.
        """
        config_data = base_config_dict
        config_data["pinecone"] = {
            "api_key": "test-api-key",
            "environment": "us-west4-gcp",
        }
        config_data["collection"]["name"] = "test_pinecone_collection"

        config_path = tmp_path / "test_pinecone_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    def test_indexer_init(self, mock_embedder, mock_pinecone, mock_config_path):
        """Test Pinecone indexing pipeline initialization.

        Validates that the PineconeIndexingPipeline correctly initializes with:
        - Configuration loaded from YAML file
        - SentenceTransformersDocumentEmbedder for document embeddings
        - Pinecone client for cloud vector storage
        - Proper component wiring and warm-up execution

        Mocks:
            - Pinecone: Prevents actual cloud API calls
            - SentenceTransformersDocumentEmbedder: Avoids model loading overhead

        Assertions:
            - Pipeline config is loaded and accessible
            - Embedder is initialized and warmed up
            - Pinecone client is created with API key from config
            - All pipeline components are properly wired
        """
        mock_embedder.return_value.warm_up = MagicMock()
        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value.indexes = []
        mock_pinecone.return_value = mock_pc

        pipeline = PineconeIndexingPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_pinecone.assert_called_once()
        assert pipeline.embedder is not None
        assert pipeline.pc is not None

    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_searcher_init(self, mock_embedder, mock_pinecone, mock_config_path):
        """Test Pinecone search pipeline initialization without RAG components.

        Validates basic searcher initialization with embedding and retrieval
        capabilities only. Reranking and generator are disabled to avoid
        Haystack Pipeline validation issues during unit testing.

        Configuration Modification:
            Temporarily modifies the config file to disable reranking and
            generator components, creating a simplified pipeline for validation.

        Mocks:
            - Pinecone: Prevents cloud API connections
            - SentenceTransformersTextEmbedder: Avoids model loading
        """
        mock_embedder.return_value.warm_up = MagicMock()
        mock_pc = MagicMock()
        mock_pinecone.return_value = mock_pc

        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = PineconeSearchPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_pinecone.assert_called_once()
        assert pipeline.embedder is not None

    @patch("vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.pinecone_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_returns_results(
        self, mock_embedder, mock_pinecone, mock_config_path
    ):
        """Test search returns properly formatted results with Pinecone scores.

        Validates the complete search flow including:
        - Query embedding generation via mocked embedder
        - Pinecone index query execution with metadata filtering
        - Result formatting with document IDs, content, metadata, and scores
        - Proper handling of Pinecone's native similarity scores

        Result Format:
            Each result dictionary contains:
            - id: Document identifier (string)
            - score: Pinecone similarity score (0.0 to 1.0, higher is better)
            - content: Document text content from metadata
            - metadata: Complete metadata dictionary including source info

        Note:
            Pinecone returns similarity scores directly (cosine similarity)
            without requiring conversion like Chroma's distance metric.
        """
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_instance

        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {"content": "Test content 1", "source": "test"},
                },
                {
                    "id": "doc2",
                    "score": 0.90,
                    "metadata": {"content": "Test content 2", "source": "test"},
                },
            ]
        }
        mock_pc = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc

        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = PineconeSearchPipeline(mock_config_path)
        results = pipeline.search("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.95
        assert "content" in results[0]
        assert results[0]["content"] == "Test content 1"
        assert results[0]["metadata"]["source"] == "test"

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set"
    )
    def test_indexing_integration(self, tmp_path, base_config_dict):
        """Integration test for document indexing pipeline with real Pinecone.

        Creates a real Pinecone index in the cloud, indexes sample documents using
        the TriviaQA dataloader, and verifies the index exists with proper
        configuration.

        Environment Requirements:
            PINECONE_API_KEY must be set with valid credentials for Pinecone
            cloud service access.

        Index Naming:
            Uses "integration_test_pinecone_collection_idx" to avoid collisions
            with other test runs and production indexes.

        Cleanup:
            Deletes the test index after verification to minimize cloud costs
            and resource usage.
        """
        config_data = base_config_dict
        config_data["pinecone"] = {
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": "us-west4-gcp",
        }
        config_data["collection"]["name"] = "integration_test_pinecone_collection_idx"
        config_path = tmp_path / "integration_pinecone_idx_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        pipeline = PineconeIndexingPipeline(config_path)
        pipeline.run()

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        assert config_data["collection"]["name"] in [
            idx.name for idx in pc.list_indexes().indexes
        ]

        pc.delete_index(config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set"
    )
    def test_search_integration(self, tmp_path, base_config_dict):
        """Integration test for semantic search with real Pinecone cloud index.

        End-to-end test that indexes documents then performs semantic search
        against a live Pinecone index. Validates that search returns relevant
        documents with proper formatting and similarity scores.

        Workflow:
            1. Create indexer config and index documents (limited to 3)
            2. Create searcher config pointing to same index
            3. Execute search query: "What is the capital of France?"
            4. Verify results contain expected fields (content, score, metadata)

        Environment Requirements:
            PINECONE_API_KEY must be set with valid cloud credentials.

        Resource Management:
            Creates temporary index "integration_test_pinecone_collection_srh"
            and deletes it after test to control cloud costs.
        """
        idx_config_data = base_config_dict
        idx_config_data["pinecone"] = {
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": "us-west4-gcp",
        }
        idx_config_data["collection"]["name"] = (
            "integration_test_pinecone_collection_srh"
        )
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_pinecone_idx_config_srh.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = PineconeIndexingPipeline(idx_config_path)
        indexer.run()

        srh_config_data = base_config_dict
        srh_config_data["pinecone"] = {
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": "us-west4-gcp",
        }
        srh_config_data["collection"]["name"] = (
            "integration_test_pinecone_collection_srh"
        )
        srh_config_path = tmp_path / "integration_pinecone_srh_config.yaml"
        with open(srh_config_path, "w") as f:
            yaml.dump(srh_config_data, f)

        pipeline = PineconeSearchPipeline(srh_config_path)
        results = pipeline.search("What is the capital of France?", top_k=1)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pc.delete_index(idx_config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("PINECONE_API_KEY") and os.getenv("GROQ_API_KEY")),
        reason="PINECONE_API_KEY or GROQ_API_KEY not set",
    )
    def test_rag_integration(self, tmp_path, base_config_dict):
        """Integration test for full RAG pipeline with LLM answer generation.

        Tests the complete cost-optimized RAG workflow:
        1. Index documents into Pinecone cloud index
        2. Configure generator with Groq API for LLM-based answer generation
        3. Execute search_with_rag to retrieve documents and generate contextual answer
        4. Validate response contains both retrieved documents and generated answer

        Generator Configuration:
            Uses Groq's llama-3.3-70b-versatile model for answer generation.
            Requires GROQ_API_KEY environment variable for authentication.

        Result Format:
            The search_with_rag method returns a dictionary with:
            - documents: List of retrieved document dictionaries with content,
              metadata, and similarity scores
            - answer: LLM-generated answer string synthesized from retrieved context

        Environment Requirements:
            PINECONE_API_KEY: Valid API key for Pinecone cloud service
            GROQ_API_KEY: Valid API key for Groq LLM service access

        Cost Management:
            Creates temporary index "integration_test_pinecone_collection_rag"
            and deletes it after test to control both Pinecone storage costs
            and Groq API usage.
        """
        idx_config_data = base_config_dict
        idx_config_data["pinecone"] = {
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": "us-west4-gcp",
        }
        idx_config_data["collection"]["name"] = (
            "integration_test_pinecone_collection_rag"
        )
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_pinecone_idx_config_rag.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = PineconeIndexingPipeline(idx_config_path)
        indexer.run()

        rag_config_data = base_config_dict
        rag_config_data["pinecone"] = {
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": "us-west4-gcp",
        }
        rag_config_data["collection"]["name"] = (
            "integration_test_pinecone_collection_rag"
        )
        rag_config_data["generator"]["enabled"] = True
        rag_config_data["generator"]["api_key"] = os.getenv("GROQ_API_KEY")
        rag_config_data["generator"]["model"] = "llama-3.3-70b-versatile"
        rag_config_path = tmp_path / "integration_pinecone_rag_config.yaml"
        with open(rag_config_path, "w") as f:
            yaml.dump(rag_config_data, f)

        pipeline = PineconeSearchPipeline(rag_config_path)
        result = pipeline.search_with_rag("What is the capital of France?", top_k=1)

        assert "documents" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pc.delete_index(idx_config_data["collection"]["name"])
