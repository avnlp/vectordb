"""Comprehensive unit tests for Weaviate multi-tenancy search pipeline.

This module tests the WeaviateMultitenancySearchPipeline class with focus on:
- Initialization with various config formats
- Connection handling and client setup
- Vector search with tenant isolation
- RAG pipeline execution
- Timing metrics creation
- Error handling scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.weaviate.search import (
    WeaviateMultitenancySearchPipeline,
)


class TestWeaviateMultitenancySearchPipeline:
    """Test suite for WeaviateMultitenancySearchPipeline."""

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.load_config")
    def test_initialization_with_config_path(
        self,
        mock_load_config: MagicMock,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline initialization with config file path."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
  api_key: test-key
collection:
  name: TestCollection
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_config = {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "test-key",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "tenant": {"id": "test-tenant"},
        }
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with config path
        pipeline = WeaviateMultitenancySearchPipeline(str(config_file))

        # Assertions
        assert pipeline.config == mock_config
        assert pipeline.tenant_context.tenant_id == "test-tenant"
        mock_load_config.assert_called_once_with(str(config_file))

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_initialization_with_dict_config(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test pipeline initialization with dictionary config."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "test-key",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "tenant": {"id": "dict-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with dict config
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline.config == config
        assert pipeline.tenant_context.tenant_id == "dict-tenant"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_rag_pipeline")
    def test_initialization_with_rag_enabled(
        self,
        mock_create_rag_pipeline: MagicMock,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test RAG pipeline setup when enabled."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "test-key",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": True},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        mock_rag_pipeline = MagicMock()
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        # Create pipeline with RAG enabled
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline._rag_pipeline is not None
        mock_create_rag_pipeline.assert_called_once_with(config)

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_initialization_with_rag_disabled(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test no RAG setup when disabled."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "test-key",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": False},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with RAG disabled
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline._rag_pipeline is None

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_initialization_with_tenant_context(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test pipeline initialization with explicit tenant context."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create explicit tenant context
        tenant_context = TenantContext(tenant_id="explicit-tenant")

        # Create pipeline with tenant context
        pipeline = WeaviateMultitenancySearchPipeline(config, tenant_context)

        # Assertions
        assert pipeline.tenant_context.tenant_id == "explicit-tenant"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_initialization_with_api_key(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test that API key authentication is configured."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "my-secret-key",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        WeaviateMultitenancySearchPipeline(config)

        # Assertions - verify client was created with auth
        mock_weaviate_client.assert_called_once()
        call_kwargs = mock_weaviate_client.call_args.kwargs
        assert call_kwargs["url"] == "http://localhost:8080"
        assert call_kwargs["auth_client_secret"] is not None

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_connect_initializes_components(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test that _connect initializes Weaviate client and embedder."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline._client == mock_client
        assert pipeline._embedder == mock_embedder
        mock_embedder.warm_up.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_connect_uses_environment_url(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that _connect uses WEAVIATE_URL from environment."""
        monkeypatch.setenv("WEAVIATE_URL", "http://env:8080")

        config: dict[str, Any] = {
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        WeaviateMultitenancySearchPipeline(config)

        # Assertions
        call_kwargs = mock_weaviate_client.call_args.kwargs
        assert call_kwargs["url"] == "http://env:8080"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_query_vector_search_success(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test successful vector search."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query results
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "TestCollection": [
                        {"content": "Test doc", "tenant_id": "test-tenant"}
                    ]
                }
            }
        }
        mock_client.query.get.return_value = mock_query

        # Create pipeline and query
        pipeline = WeaviateMultitenancySearchPipeline(config)
        result = pipeline.query("test query", top_k=5)

        # Assertions
        assert result.tenant_id == "test-tenant"
        assert result.query == "test query"
        assert len(result.documents) == 1
        assert result.documents[0].content == "Test doc"
        assert len(result.scores) == 1

        # Verify tenant filtering
        mock_query.with_tenant.assert_called_once_with("test-tenant")

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_query_with_tenant_namespace(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test namespace filtering per tenant."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "tenant-a"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query chain
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {"data": {"Get": {"TestCollection": []}}}
        mock_client.query.get.return_value = mock_query

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Query with explicit tenant_id
        result = pipeline.query("test query", tenant_id="tenant-b")

        # Assertions
        assert result.tenant_id == "tenant-b"
        mock_query.with_tenant.assert_called_with("tenant-b")

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_query_with_top_k_parameter(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test top_k parameter handling."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query chain
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {"data": {"Get": {"TestCollection": []}}}
        mock_client.query.get.return_value = mock_query

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Query with custom top_k
        pipeline.query("test query", top_k=20)

        # Assertions
        mock_query.with_limit.assert_called_once_with(20)

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_query_with_output_dimension_truncation(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test embedding truncation when output_dimension specified."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {
                "dimension": 384,
                "output_dimension": 256,
            },
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Return 384-dim embedding
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query chain
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {"data": {"Get": {"TestCollection": []}}}
        mock_client.query.get.return_value = mock_query

        # Create pipeline and query
        pipeline = WeaviateMultitenancySearchPipeline(config)
        pipeline.query("test query")

        # Assertions - verify truncated embedding passed to near_vector
        call_args = mock_query.with_near_vector.call_args.args[0]
        assert len(call_args["vector"]) == 256

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_query_with_empty_results(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test query with empty results."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock empty results
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {"data": {"Get": {"TestCollection": []}}}
        mock_client.query.get.return_value = mock_query

        # Create pipeline and query
        pipeline = WeaviateMultitenancySearchPipeline(config)
        result = pipeline.query("test query")

        # Assertions
        assert len(result.documents) == 0
        assert len(result.scores) == 0
        assert result.tenant_id == "test-tenant"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_query_with_missing_data_key(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test query handling when 'data' key is missing."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock result without 'data' key
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {}  # No 'data' key
        mock_client.query.get.return_value = mock_query

        # Create pipeline and query
        pipeline = WeaviateMultitenancySearchPipeline(config)
        result = pipeline.query("test query")

        # Assertions - should handle gracefully
        assert len(result.documents) == 0

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_rag_pipeline")
    def test_rag_pipeline_execution(
        self,
        mock_create_rag_pipeline: MagicMock,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test RAG pipeline execution."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": True},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query results
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "TestCollection": [
                        {"content": "Context doc", "tenant_id": "test-tenant"}
                    ]
                }
            }
        }
        mock_client.query.get.return_value = mock_query

        # Mock RAG pipeline
        mock_rag_pipeline = MagicMock()
        mock_rag_pipeline.run.return_value = {
            "generator": {"replies": ["Generated answer"]}
        }
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        # Create pipeline and run RAG
        pipeline = WeaviateMultitenancySearchPipeline(config)
        result = pipeline.rag("What is AI?", top_k=5)

        # Assertions
        assert result.tenant_id == "test-tenant"
        assert result.query == "What is AI?"
        assert result.generated_response == "Generated answer"
        assert len(result.retrieved_documents) == 1

        # Verify RAG pipeline called correctly
        mock_rag_pipeline.run.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_rag_raises_error_when_disabled(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test error when RAG not configured."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": False},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Verify error raised
        with pytest.raises(ValueError, match="RAG pipeline not enabled"):
            pipeline.rag("test query")

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_rag_pipeline")
    def test_rag_with_explicit_tenant(
        self,
        mock_create_rag_pipeline: MagicMock,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test RAG with explicit tenant_id parameter."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "tenant-a"},
            "rag": {"enabled": True},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query results
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.return_value = {
            "data": {
                "Get": {
                    "TestCollection": [{"content": "Context", "tenant_id": "tenant-b"}]
                }
            }
        }
        mock_client.query.get.return_value = mock_query

        mock_rag_pipeline = MagicMock()
        mock_rag_pipeline.run.return_value = {"generator": {"replies": ["Answer"]}}
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        # Create pipeline and run RAG with explicit tenant
        pipeline = WeaviateMultitenancySearchPipeline(config)
        result = pipeline.rag("test query", tenant_id="tenant-b")

        # Assertions
        assert result.tenant_id == "tenant-b"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_create_timing_metrics(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test metrics creation."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Test metrics creation
        metrics = pipeline._create_timing_metrics(150.5)

        # Assertions
        assert metrics.tenant_resolution_ms == 0.0
        assert metrics.index_operation_ms == 0.0
        assert metrics.retrieval_ms == 150.5
        assert metrics.total_ms == 150.5
        assert metrics.tenant_id == "test-tenant"
        assert metrics.num_documents == 0

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_get_class_name(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test _get_class_name returns configured class name."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "CustomCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline._get_class_name() == "CustomCollection"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_close_method(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test close method calls client.close()."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline and close
        pipeline = WeaviateMultitenancySearchPipeline(config)
        pipeline.close()

        # Assertions
        mock_client.close.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_error_handling_on_connection_failure(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test error handling on connection failure."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "test-key",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks to raise exception
        mock_weaviate_client.side_effect = Exception("Connection failed")

        # Verify error raised during initialization
        with pytest.raises(Exception, match="Connection failed"):
            WeaviateMultitenancySearchPipeline(config)

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_error_handling_on_query_failure(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test error handling on query failure."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query chain to raise exception
        mock_query = MagicMock()
        mock_query.get.return_value = mock_query
        mock_query.with_tenant.return_value = mock_query
        mock_query.with_near_vector.return_value = mock_query
        mock_query.with_limit.return_value = mock_query
        mock_query.do.side_effect = Exception("Query failed")
        mock_client.query.get.return_value = mock_query

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Verify error raised during query
        with pytest.raises(Exception, match="Query failed"):
            pipeline.query("test query")

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_initialization_with_additional_headers(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test that additional headers are passed to client."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
                "headers": {"X-OpenAI-Api-Key": "test-key"},
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        WeaviateMultitenancySearchPipeline(config)

        # Assertions - verify additional_headers was set
        call_kwargs = mock_weaviate_client.call_args.kwargs
        assert "additional_headers" in call_kwargs
        assert call_kwargs["additional_headers"]["X-OpenAI-Api-Key"] == "test-key"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_close_method_when_client_is_none(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
    ) -> None:
        """Test close method handles None client gracefully."""
        config: dict[str, Any] = {
            "weaviate": {
                "url": "http://localhost:8080",
            },
            "collection": {"name": "TestCollection"},
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancySearchPipeline(config)

        # Manually set _client to None to test the edge case
        pipeline._client = None

        # close() should not raise even when client is None
        pipeline.close()  # Should not raise
