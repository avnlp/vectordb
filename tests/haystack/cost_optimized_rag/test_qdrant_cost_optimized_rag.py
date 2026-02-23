"""Tests for Qdrant cost-optimized RAG pipeline."""

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml
from qdrant_client import QdrantClient

from vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer import (
    QdrantIndexingPipeline,
)
from vectordb.haystack.cost_optimized_rag.search.qdrant_searcher import (
    QdrantSearchPipeline,
)


class TestQdrantCostOptimizedRag:
    """Unit and integration tests for Qdrant cost-optimized RAG."""

    @pytest.fixture
    def mock_config_path(self, tmp_path, base_config_dict):
        """Create a temporary test config file."""
        config_data = base_config_dict
        config_data["qdrant"] = {
            "host": "localhost",
            "port": 6333,
            "api_key": "",
            "https": False,
        }
        config_data["collection"]["name"] = "test_qdrant_collection"

        config_path = tmp_path / "test_qdrant_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    def test_indexer_init(self, mock_embedder, mock_qdrant_client, mock_config_path):
        """Test indexer initialization."""
        mock_embedder.return_value.warm_up = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.get_collection.side_effect = (
            Exception  # Simulate collection not existing
        )
        mock_qdrant_client.return_value = mock_client_instance

        pipeline = QdrantIndexingPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_qdrant_client.assert_called_once()
        assert pipeline.embedder is not None
        assert pipeline.client is not None

    @patch("vectordb.haystack.cost_optimized_rag.search.qdrant_searcher.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.qdrant_searcher.SentenceTransformersTextEmbedder"
    )
    def test_searcher_init(self, mock_embedder, mock_qdrant_client, mock_config_path):
        """Test searcher initialization without RAG components."""
        mock_embedder.return_value.warm_up = MagicMock()
        mock_client_instance = MagicMock()
        mock_qdrant_client.return_value = mock_client_instance

        # Disable reranking and generator to avoid Haystack Pipeline validation issues
        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = QdrantSearchPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_qdrant_client.assert_called_once()
        assert pipeline.embedder is not None

    @patch("vectordb.haystack.cost_optimized_rag.search.qdrant_searcher.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.qdrant_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_returns_results(
        self, mock_embedder, mock_qdrant_client, mock_config_path
    ):
        """Test search returns properly formatted results."""
        from qdrant_client.models import ScoredPoint

        # Setup mocks
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_instance

        mock_scored_point1 = ScoredPoint(
            id="doc1",
            version=1,
            score=0.95,
            payload={"content": "Test content 1", "source": "test"},
            vector=None,
        )
        mock_scored_point2 = ScoredPoint(
            id="doc2",
            version=1,
            score=0.90,
            payload={"content": "Test content 2", "source": "test"},
            vector=None,
        )
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = [
            mock_scored_point1,
            mock_scored_point2,
        ]
        mock_qdrant_client.return_value = mock_client_instance

        # Disable reranking and generator for this specific test
        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = QdrantSearchPipeline(mock_config_path)
        results = pipeline.search("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.95
        assert "content" in results[0]
        assert results[0]["content"] == "Test content 1"
        assert results[0]["metadata"]["source"] == "test"
        mock_client_instance.search.assert_called_once()

    # NOTE: Qdrant integration tests require a running Qdrant instance.
    #       They are skipped by default and require QDRANT_HOST and QDRANT_PORT
    #       environment variables to be set.

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("QDRANT_HOST") and os.getenv("QDRANT_PORT")),
        reason="QDRANT_HOST and QDRANT_PORT not set for Qdrant integration tests",
    )
    def test_indexing_integration(self, tmp_path, base_config_dict):
        """Integration test for indexing pipeline."""
        config_data = base_config_dict
        config_data["qdrant"] = {
            "host": os.getenv("QDRANT_HOST"),
            "port": int(os.getenv("QDRANT_PORT")),
            "api_key": os.getenv("QDRANT_API_KEY", ""),
            "https": os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        }
        config_data["collection"]["name"] = "integration_test_qdrant_idx"
        config_data["dataloader"]["limit"] = 3
        config_path = tmp_path / "integration_qdrant_idx_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        pipeline = QdrantIndexingPipeline(config_path)
        pipeline.run()

        # Verify collection exists (basic check)
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=int(os.getenv("QDRANT_PORT")),
            api_key=os.getenv("QDRANT_API_KEY", ""),
            https=os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        )
        assert client.get_collection(config_data["collection"]["name"]) is not None

        # Clean up
        client.delete_collection(config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("QDRANT_HOST") and os.getenv("QDRANT_PORT")),
        reason="QDRANT_HOST and QDRANT_PORT not set for Qdrant integration tests",
    )
    def test_search_integration(self, tmp_path, base_config_dict):
        """Integration test for search pipeline."""
        # Index documents first
        idx_config_data = base_config_dict
        idx_config_data["qdrant"] = {
            "host": os.getenv("QDRANT_HOST"),
            "port": int(os.getenv("QDRANT_PORT")),
            "api_key": os.getenv("QDRANT_API_KEY", ""),
            "https": os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        }
        idx_config_data["collection"]["name"] = "integration_test_qdrant_srh"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_qdrant_idx_config_srh.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = QdrantIndexingPipeline(idx_config_path)
        indexer.run()

        # Now search
        srh_config_data = base_config_dict
        srh_config_data["qdrant"] = {
            "host": os.getenv("QDRANT_HOST"),
            "port": int(os.getenv("QDRANT_PORT")),
            "api_key": os.getenv("QDRANT_API_KEY", ""),
            "https": os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        }
        srh_config_data["collection"]["name"] = "integration_test_qdrant_srh"
        srh_config_path = tmp_path / "integration_qdrant_srh_config.yaml"
        with open(srh_config_path, "w") as f:
            yaml.dump(srh_config_data, f)

        pipeline = QdrantSearchPipeline(srh_config_path)
        results = pipeline.search("What is the capital of France?", top_k=1)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

        # Clean up
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=int(os.getenv("QDRANT_PORT")),
            api_key=os.getenv("QDRANT_API_KEY", ""),
            https=os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        )
        client.delete_collection(idx_config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (
            os.getenv("QDRANT_HOST")
            and os.getenv("QDRANT_PORT")
            and os.getenv("GROQ_API_KEY")
        ),
        reason="QDRANT_HOST, QDRANT_PORT, or GROQ_API_KEY not set for Qdrant RAG integration tests",
    )
    def test_rag_integration(self, tmp_path, base_config_dict):
        """Integration test for RAG pipeline."""
        # Index documents first
        idx_config_data = base_config_dict
        idx_config_data["qdrant"] = {
            "host": os.getenv("QDRANT_HOST"),
            "port": int(os.getenv("QDRANT_PORT")),
            "api_key": os.getenv("QDRANT_API_KEY", ""),
            "https": os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        }
        idx_config_data["collection"]["name"] = "integration_test_qdrant_rag"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_qdrant_idx_config_rag.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = QdrantIndexingPipeline(idx_config_path)
        indexer.run()

        # Now search with RAG
        rag_config_data = base_config_dict
        rag_config_data["qdrant"] = {
            "host": os.getenv("QDRANT_HOST"),
            "port": int(os.getenv("QDRANT_PORT")),
            "api_key": os.getenv("QDRANT_API_KEY", ""),
            "https": os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        }
        rag_config_data["collection"]["name"] = "integration_test_qdrant_rag"
        rag_config_data["generator"]["enabled"] = True
        rag_config_data["generator"]["api_key"] = os.getenv("GROQ_API_KEY")
        rag_config_data["generator"]["model"] = (
            "llama-3.3-70b-versatile"  # Ensure a valid model is set for Groq
        )
        rag_config_path = tmp_path / "integration_qdrant_rag_config.yaml"
        with open(rag_config_path, "w") as f:
            yaml.dump(rag_config_data, f)

        pipeline = QdrantSearchPipeline(rag_config_path)
        result = pipeline.search_with_rag("What is the capital of France?", top_k=1)

        assert "documents" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Clean up
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST"),
            port=int(os.getenv("QDRANT_PORT")),
            api_key=os.getenv("QDRANT_API_KEY", ""),
            https=os.getenv("QDRANT_HTTPS", "False").lower() == "true",
        )
        client.delete_collection(idx_config_data["collection"]["name"])
