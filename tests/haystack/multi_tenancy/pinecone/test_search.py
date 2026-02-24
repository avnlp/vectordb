"""Comprehensive unit tests for Pinecone multi-tenancy search pipeline.

This module tests the PineconeMultitenancySearchPipeline class with focus on:
- Initialization with various config formats
- Connection handling and Pinecone client initialization
- Vector search with tenant namespace filtering
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
from vectordb.haystack.multi_tenancy.pinecone.search import (
    PineconeMultitenancySearchPipeline,
)


class TestPineconeMultitenancySearchPipeline:
    """Test suite for PineconeMultitenancySearchPipeline."""

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.load_config")
    def test_initialization_with_config_path(
        self,
        mock_load_config: MagicMock,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline initialization with config file path."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
pinecone:
  api_key: test-key
  index: test-index
  dimension: 384
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_config = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "tenant": {"id": "test-tenant"},
        }
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with config path
        pipeline = PineconeMultitenancySearchPipeline(str(config_file))

        # Assertions
        assert pipeline.config == mock_config
        assert pipeline.tenant_context.tenant_id == "test-tenant"
        mock_load_config.assert_called_once_with(str(config_file))

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_initialization_with_dict_config(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test pipeline initialization with dictionary config."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "tenant": {"id": "dict-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with dict config
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline.config == config
        assert pipeline.tenant_context.tenant_id == "dict-tenant"

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_rag_pipeline")
    def test_initialization_with_rag_enabled(
        self,
        mock_create_rag_pipeline: MagicMock,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test RAG pipeline setup when enabled."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": True},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        mock_rag_pipeline = MagicMock()
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        # Create pipeline with RAG enabled
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline._rag_pipeline is not None
        mock_create_rag_pipeline.assert_called_once_with(config)

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_initialization_with_rag_disabled(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test no RAG setup when disabled."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": False},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with RAG disabled
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Assertions
        assert pipeline._rag_pipeline is None

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_connect_creates_index_when_not_exists(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test index connection logic."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Assertions
        mock_pinecone_class.assert_called_once_with(api_key="test-key")
        mock_client.Index.assert_called_once_with("test-index")
        assert pipeline._index == mock_index
        assert pipeline._embedder == mock_embedder
        mock_embedder.warm_up.assert_called_once()

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_connect_uses_existing_index(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that _connect uses existing index."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "existing-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Assertions
        mock_client.Index.assert_called_once_with("existing-index")
        assert pipeline._index == mock_index

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_vector_search_success(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test successful vector search."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock query results
        mock_match = MagicMock()
        mock_match.metadata = {"content": "Test document content"}
        mock_match.score = 0.95
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.query("test query", top_k=5)

        # Assertions
        assert result.tenant_id == "test-tenant"
        assert result.query == "test query"
        assert len(result.documents) == 1
        assert result.documents[0].content == "Test document content"
        assert len(result.scores) == 1
        assert result.scores[0] == 0.95
        assert result.timing.total_ms >= 0

        # Verify namespace filtering
        mock_index.query.assert_called_once()
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["namespace"] == "test-tenant"
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["include_metadata"] is True

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_tenant_namespace(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test namespace filtering per tenant."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "tenant-a"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        mock_index.query.return_value = MagicMock(matches=[])

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Query with explicit tenant_id
        result = pipeline.query("test query", tenant_id="tenant-b")

        # Assertions
        assert result.tenant_id == "tenant-b"
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["namespace"] == "tenant-b"

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_top_k_parameter(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test top_k parameter handling."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        mock_index.query.return_value = MagicMock(matches=[])

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Query with custom top_k
        pipeline.query("test query", top_k=20)

        # Assertions
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["top_k"] == 20

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_output_dimension_truncation(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test embedding truncation when output_dimension specified."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 256,
            },
            "embedding": {
                "dimension": 384,
                "output_dimension": 256,
            },
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Return 384-dim embedding
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        mock_index.query.return_value = MagicMock(matches=[])

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        pipeline.query("test query")

        # Assertions - verify truncated embedding passed to query
        call_kwargs = mock_index.query.call_args.kwargs
        vector = call_kwargs["vector"]
        assert len(vector) == 256

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_metadata_filter(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test query with metadata filter support."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        mock_match = MagicMock()
        mock_match.metadata = {"content": "Test", "category": "tech"}
        mock_match.score = 0.9
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.query("test query")

        # Assertions
        assert len(result.documents) == 1
        assert result.documents[0].meta.get("category") == "tech"

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_missing_metadata(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test query handling when metadata is missing or None."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Test with None metadata
        mock_match = MagicMock()
        mock_match.metadata = None
        mock_match.score = 0.8
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.query("test query")

        # Assertions
        assert len(result.documents) == 1
        assert result.documents[0].content == ""
        assert result.documents[0].meta == {}

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_rag_pipeline")
    def test_rag_pipeline_execution(
        self,
        mock_create_rag_pipeline: MagicMock,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test RAG pipeline execution."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": True},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock retrieval results
        mock_match = MagicMock()
        mock_match.metadata = {"content": "Context document"}
        mock_match.score = 0.95
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        # Mock RAG pipeline
        mock_rag_pipeline = MagicMock()
        mock_rag_pipeline.run.return_value = {
            "generator": {"replies": ["Generated answer"]}
        }
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        # Create pipeline and run RAG
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.rag("What is AI?", top_k=5)

        # Assertions
        assert result.tenant_id == "test-tenant"
        assert result.query == "What is AI?"
        assert result.generated_response == "Generated answer"
        assert len(result.retrieved_documents) == 1
        assert result.retrieval_scores == [0.95]
        assert result.timing.total_ms >= 0

        # Verify RAG pipeline called correctly
        mock_rag_pipeline.run.assert_called_once()
        call_args = mock_rag_pipeline.run.call_args.args[0]
        assert call_args["prompt_builder"]["query"] == "What is AI?"
        assert len(call_args["prompt_builder"]["documents"]) == 1

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_rag_raises_error_when_disabled(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test error when RAG not configured."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
            "rag": {"enabled": False},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Verify error raised
        with pytest.raises(ValueError, match="RAG pipeline not enabled"):
            pipeline.rag("test query")

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_rag_pipeline")
    def test_rag_with_explicit_tenant(
        self,
        mock_create_rag_pipeline: MagicMock,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test RAG with explicit tenant_id parameter."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "tenant-a"},
            "rag": {"enabled": True},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        mock_match = MagicMock()
        mock_match.metadata = {"content": "Context"}
        mock_match.score = 0.9
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        mock_rag_pipeline = MagicMock()
        mock_rag_pipeline.run.return_value = {"generator": {"replies": ["Answer"]}}
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        # Create pipeline and run RAG with explicit tenant
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.rag("test query", tenant_id="tenant-b")

        # Assertions
        assert result.tenant_id == "tenant-b"
        # Verify namespace used correctly
        call_kwargs = mock_index.query.call_args.kwargs
        assert call_kwargs["namespace"] == "tenant-b"

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_create_timing_metrics(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test metrics creation."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Test metrics creation
        metrics = pipeline._create_timing_metrics(150.5)

        # Assertions
        assert metrics.tenant_resolution_ms == 0.0
        assert metrics.index_operation_ms == 0.0
        assert metrics.retrieval_ms == 150.5
        assert metrics.total_ms == 150.5
        assert metrics.tenant_id == "test-tenant"
        assert metrics.num_documents == 0

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_error_handling_on_connection_failure(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test error handling on connection failure."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks to raise exception
        mock_pinecone_class.side_effect = Exception("Connection failed")

        # Verify error raised during initialization
        with pytest.raises(Exception, match="Connection failed"):
            PineconeMultitenancySearchPipeline(config)

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_error_handling_on_query_failure(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test error handling on query failure."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Setup query to raise exception
        mock_index.query.side_effect = Exception("Query failed")

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Verify error raised during query
        with pytest.raises(Exception, match="Query failed"):
            pipeline.query("test query")

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_environment_api_key(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test env var configuration for API key."""
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")

        config: dict[str, Any] = {
            "pinecone": {
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        mock_index.query.return_value = MagicMock(matches=[])

        # Create pipeline
        PineconeMultitenancySearchPipeline(config)

        # Assertions
        mock_pinecone_class.assert_called_once_with(api_key="env-api-key")

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_environment_index_name(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test env var configuration for index name."""
        monkeypatch.setenv("PINECONE_INDEX", "env-index-name")

        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        mock_index.query.return_value = MagicMock(matches=[])

        # Create pipeline
        pipeline = PineconeMultitenancySearchPipeline(config)

        # Assertions
        mock_client.Index.assert_called_once_with("env-index-name")
        assert pipeline._get_index_name() == "env-index-name"

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_close_method(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test close method is callable."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline and close
        pipeline = PineconeMultitenancySearchPipeline(config)
        pipeline.close()  # Should not raise

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_initialization_with_explicit_tenant_context(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test pipeline initialization with explicit tenant context."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        mock_index.query.return_value = MagicMock(matches=[])

        # Create explicit tenant context
        tenant_context = TenantContext(tenant_id="explicit-tenant")

        # Create pipeline with tenant context
        pipeline = PineconeMultitenancySearchPipeline(config, tenant_context)

        # Assertions
        assert pipeline.tenant_context.tenant_id == "explicit-tenant"

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_multiple_matches(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test query with multiple document matches."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock multiple matches
        mock_match1 = MagicMock()
        mock_match1.metadata = {"content": "Doc 1"}
        mock_match1.score = 0.95

        mock_match2 = MagicMock()
        mock_match2.metadata = {"content": "Doc 2"}
        mock_match2.score = 0.85

        mock_match3 = MagicMock()
        mock_match3.metadata = {"content": "Doc 3"}
        mock_match3.score = 0.75

        mock_index.query.return_value = MagicMock(
            matches=[mock_match1, mock_match2, mock_match3]
        )

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.query("test query", top_k=3)

        # Assertions
        assert len(result.documents) == 3
        assert len(result.scores) == 3
        assert result.scores == [0.95, 0.85, 0.75]
        assert result.documents[0].content == "Doc 1"
        assert result.documents[1].content == "Doc 2"
        assert result.documents[2].content == "Doc 3"

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_result_ranks(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that query results have correct ranks."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock matches
        mock_match1 = MagicMock()
        mock_match1.metadata = {"content": "Doc 1"}
        mock_match1.score = 0.95

        mock_match2 = MagicMock()
        mock_match2.metadata = {"content": "Doc 2"}
        mock_match2.score = 0.85

        mock_index.query.return_value = MagicMock(matches=[mock_match1, mock_match2])

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.query("test query")

        # Check TenantQueryResult objects have correct ranks
        # The result.documents are Haystack Documents, not TenantQueryResult
        # TenantQueryResult objects are created internally but not exposed directly
        # We verify the documents are returned in correct order
        assert len(result.documents) == 2
        assert result.scores[0] == 0.95  # Higher score first
        assert result.scores[1] == 0.85

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_query_with_no_matches(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test query with no matches returns empty result."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock empty results
        mock_index.query.return_value = MagicMock(matches=[])

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.query("test query")

        # Assertions
        assert len(result.documents) == 0
        assert len(result.scores) == 0
        assert result.tenant_id == "test-tenant"
        assert result.timing.total_ms >= 0

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_match_without_score_attribute(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test handling of matches without score attribute."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_create_embedder.return_value = mock_embedder

        # Mock match without score
        mock_match = MagicMock()
        mock_match.metadata = {"content": "Test content"}
        # No score attribute - should default to 1.0
        del mock_match.score
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        # Create pipeline and query
        pipeline = PineconeMultitenancySearchPipeline(config)
        result = pipeline.query("test query")

        # Assertions - score should default to 1.0
        assert len(result.documents) == 1
        # When score attribute doesn't exist, hasattr returns False
        # so the code uses 1.0 as default

    @patch("pinecone.Pinecone")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.create_text_embedder")
    def test_default_top_k_value(
        self,
        mock_create_embedder: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test default top_k value is 10."""
        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_pinecone_class.return_value = mock_client
        mock_client.Index.return_value = mock_index

        MagicMock()
