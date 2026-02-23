"""Tests for Weaviate cost-optimized RAG pipeline."""

import os
from unittest.mock import MagicMock, patch

import pytest
import weaviate
import yaml

from vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer import (
    WeaviateIndexingPipeline,
)
from vectordb.haystack.cost_optimized_rag.search.weaviate_searcher import (
    WeaviateSearchPipeline,
)


class TestWeaviateCostOptimizedRag:
    """Unit and integration tests for Weaviate cost-optimized RAG."""

    @pytest.fixture
    def mock_config_path(self, tmp_path, base_config_dict):
        """Create a temporary test config file."""
        config_data = base_config_dict
        config_data["weaviate"] = {
            "host": "localhost",
            "port": 8080,
            "api_key": "",
        }
        config_data["collection"]["name"] = (
            "TestWeaviateCollection"  # Weaviate class names are capitalized
        )

        config_path = tmp_path / "test_weaviate_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate.Client"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    def test_indexer_init(self, mock_embedder, mock_weaviate_client, mock_config_path):
        """Test indexer initialization."""
        mock_embedder.return_value.warm_up = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.schema.delete_class.side_effect = (
            Exception  # Simulate class not existing
        )
        mock_weaviate_client.return_value = mock_client_instance

        pipeline = WeaviateIndexingPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_weaviate_client.assert_called_once()
        assert pipeline.embedder is not None
        assert pipeline.client is not None

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.weaviate_searcher.weaviate.Client"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.weaviate_searcher.SentenceTransformersTextEmbedder"
    )
    def test_searcher_init(self, mock_embedder, mock_weaviate_client, mock_config_path):
        """Test searcher initialization without RAG components."""
        mock_embedder.return_value.warm_up = MagicMock()
        mock_client_instance = MagicMock()
        mock_weaviate_client.return_value = mock_client_instance

        # Disable reranking and generator to avoid Haystack Pipeline validation issues
        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = WeaviateSearchPipeline(mock_config_path)

        assert pipeline.config is not None
        mock_embedder.assert_called_once()
        mock_weaviate_client.assert_called_once()
        assert pipeline.embedder is not None

    @patch(
        "vectordb.haystack.cost_optimized_rag.search.weaviate_searcher.weaviate.Client"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.search.weaviate_searcher.SentenceTransformersTextEmbedder"
    )
    def test_search_returns_results(
        self, mock_embedder, mock_weaviate_client, mock_config_path
    ):
        """Test search returns properly formatted results."""
        # Setup mocks
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder.return_value = mock_embedder_instance

        mock_query_builder = MagicMock()
        mock_query_builder.with_near_vector.return_value = mock_query_builder
        mock_query_builder.with_limit.return_value = mock_query_builder
        mock_query_builder.do.return_value = {
            "data": {
                "Get": {
                    "TestWeaviateCollection": [
                        {
                            "_id": "doc1",
                            "content": "Test content 1",
                            "metadata": '{"source": "test"}',
                        },
                        {
                            "_id": "doc2",
                            "content": "Test content 2",
                            "metadata": '{"source": "test"}',
                        },
                    ]
                }
            }
        }
        mock_client_instance = MagicMock()
        mock_client_instance.query.get.return_value = mock_query_builder
        mock_weaviate_client.return_value = mock_client_instance

        # Disable reranking and generator for this specific test
        with open(mock_config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config_data["search"]["reranking_enabled"] = False
        config_data["generator"]["enabled"] = False
        with open(mock_config_path, "w") as f:
            yaml.safe_dump(config_data, f)

        pipeline = WeaviateSearchPipeline(mock_config_path)
        results = pipeline.search("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 1.0  # Weaviate doesn't return score easily
        assert "content" in results[0]
        assert results[0]["content"] == "Test content 1"
        assert results[0]["metadata"]["source"] == "test"
        mock_client_instance.query.get.assert_called_once_with(
            "TestWeaviateCollection", ["content", "metadata", "_id"]
        )

    # NOTE: Weaviate integration tests require a running Weaviate instance.
    #       They are skipped by default and require WEAVIATE_HOST and WEAVIATE_PORT
    #       environment variables to be set.

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("WEAVIATE_HOST") and os.getenv("WEAVIATE_PORT")),
        reason="WEAVIATE_HOST and WEAVIATE_PORT not set for Weaviate integration tests",
    )
    def test_indexing_integration(self, tmp_path, base_config_dict):
        """Integration test for indexing pipeline."""
        config_data = base_config_dict
        config_data["weaviate"] = {
            "host": os.getenv("WEAVIATE_HOST"),
            "port": int(os.getenv("WEAVIATE_PORT")),
            "api_key": os.getenv("WEAVIATE_API_KEY", ""),
        }
        config_data["collection"]["name"] = "IntegrationTestWeaviateIdx"
        config_data["dataloader"]["limit"] = 3
        config_path = tmp_path / "integration_weaviate_idx_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        pipeline = WeaviateIndexingPipeline(config_path)
        pipeline.run()

        # Verify collection exists (basic check)
        client = weaviate.Client(
            url=f"http://{os.getenv('WEAVIATE_HOST')}:{os.getenv('WEAVIATE_PORT')}",
            auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
            if os.getenv("WEAVIATE_API_KEY")
            else None,
        )
        assert (
            client.schema.get()["classes"][0]["class"]
            == config_data["collection"]["name"]
        )

        # Clean up
        client.schema.delete_class(config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (os.getenv("WEAVIATE_HOST") and os.getenv("WEAVIATE_PORT")),
        reason="WEAVIATE_HOST and WEAVIATE_PORT not set for Weaviate integration tests",
    )
    def test_search_integration(self, tmp_path, base_config_dict):
        """Integration test for search pipeline."""
        # Index documents first
        idx_config_data = base_config_dict
        idx_config_data["weaviate"] = {
            "host": os.getenv("WEAVIATE_HOST"),
            "port": int(os.getenv("WEAVIATE_PORT")),
            "api_key": os.getenv("WEAVIATE_API_KEY", ""),
        }
        idx_config_data["collection"]["name"] = "IntegrationTestWeaviateSrh"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_weaviate_idx_config_srh.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = WeaviateIndexingPipeline(idx_config_path)
        indexer.run()

        # Now search
        srh_config_data = base_config_dict
        srh_config_data["weaviate"] = {
            "host": os.getenv("WEAVIATE_HOST"),
            "port": int(os.getenv("WEAVIATE_PORT")),
            "api_key": os.getenv("WEAVIATE_API_KEY", ""),
        }
        srh_config_data["collection"]["name"] = "IntegrationTestWeaviateSrh"
        srh_config_path = tmp_path / "integration_weaviate_srh_config.yaml"
        with open(srh_config_path, "w") as f:
            yaml.dump(srh_config_data, f)

        pipeline = WeaviateSearchPipeline(srh_config_path)
        results = pipeline.search("What is the capital of France?", top_k=1)

        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

        # Clean up
        client = weaviate.Client(
            url=f"http://{os.getenv('WEAVIATE_HOST')}:{os.getenv('WEAVIATE_PORT')}",
            auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
            if os.getenv("WEAVIATE_API_KEY")
            else None,
        )
        client.schema.delete_class(idx_config_data["collection"]["name"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not (
            os.getenv("WEAVIATE_HOST")
            and os.getenv("WEAVIATE_PORT")
            and os.getenv("GROQ_API_KEY")
        ),
        reason="WEAVIATE_HOST, WEAVIATE_PORT, or GROQ_API_KEY not set for Weaviate RAG integration tests",
    )
    def test_rag_integration(self, tmp_path, base_config_dict):
        """Integration test for RAG pipeline."""
        # Index documents first
        idx_config_data = base_config_dict
        idx_config_data["weaviate"] = {
            "host": os.getenv("WEAVIATE_HOST"),
            "port": int(os.getenv("WEAVIATE_PORT")),
            "api_key": os.getenv("WEAVIATE_API_KEY", ""),
        }
        idx_config_data["collection"]["name"] = "IntegrationTestWeaviateRag"
        idx_config_data["dataloader"]["limit"] = 3
        idx_config_path = tmp_path / "integration_weaviate_idx_config_rag.yaml"
        with open(idx_config_path, "w") as f:
            yaml.dump(idx_config_data, f)

        indexer = WeaviateIndexingPipeline(idx_config_path)
        indexer.run()

        # Now search with RAG
        rag_config_data = base_config_dict
        rag_config_data["weaviate"] = {
            "host": os.getenv("WEAVIATE_HOST"),
            "port": int(os.getenv("WEAVIATE_PORT")),
            "api_key": os.getenv("WEAVIATE_API_KEY", ""),
        }
        rag_config_data["collection"]["name"] = "IntegrationTestWeaviateRag"
        rag_config_data["generator"]["enabled"] = True
        rag_config_data["generator"]["api_key"] = os.getenv("GROQ_API_KEY")
        rag_config_data["generator"]["model"] = (
            "llama-3.3-70b-versatile"  # Ensure a valid model is set for Groq
        )
        rag_config_path = tmp_path / "integration_weaviate_rag_config.yaml"
        with open(rag_config_path, "w") as f:
            yaml.dump(rag_config_data, f)

        pipeline = WeaviateSearchPipeline(rag_config_path)
        result = pipeline.search_with_rag("What is the capital of France?", top_k=1)

        assert "documents" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Clean up
        client = weaviate.Client(
            url=f"http://{os.getenv('WEAVIATE_HOST')}:{os.getenv('WEAVIATE_PORT')}",
            auth_client_secret=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
            if os.getenv("WEAVIATE_API_KEY")
            else None,
        )
        client.schema.delete_class(idx_config_data["collection"]["name"])
