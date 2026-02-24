"""Tests for Pinecone multi-tenancy pipelines."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


class TestPineconeMultitenancy:
    """Test suite for Pinecone multi-tenancy pipelines."""

    def test_tenant_context_resolution(self) -> None:
        """Test tenant context resolution."""
        tenant_ctx = TenantContext(tenant_id="test_tenant")
        assert tenant_ctx.tenant_id == "test_tenant"

    def test_tenant_context_from_config(self) -> None:
        """Test tenant context resolution from config."""
        config = {"tenant": {"id": "config_tenant"}}
        tenant_ctx = TenantContext.resolve(None, config)
        assert tenant_ctx.tenant_id == "config_tenant"

    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test that Pinecone indexing pipeline initializes correctly."""
        # Add tenant config
        pinecone_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = pinecone_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.indexing import (
            PineconeMultitenancyIndexingPipeline,
        )

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(pinecone_config)
            assert pipeline.config == pinecone_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @patch("vectordb.haystack.multi_tenancy.pinecone.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test that Pinecone search pipeline initializes correctly."""
        # Add tenant config
        pinecone_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = pinecone_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.search import (
            PineconeMultitenancySearchPipeline,
        )

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(pinecone_config)
            assert pipeline.config == pinecone_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Pinecone instance)."""
        pytest.skip("Integration test requires Pinecone service")


class TestPineconeMultitenancyIndexing:
    """Extended test suite for Pinecone indexing pipeline."""

    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.load_config")
    def test_get_index_name_from_config(self, mock_load_config, mock_resolve_context):
        """Test _get_index_name returns value from config."""
        config = {
            "pinecone": {"index": "custom-index", "api_key": "test-key"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.indexing import (
            PineconeMultitenancyIndexingPipeline,
        )

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(config)
            index_name = pipeline._get_index_name()
            assert index_name == "custom-index"

    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.load_config")
    def test_get_index_name_from_env(self, mock_load_config, mock_resolve_context):
        """Test _get_index_name returns value from environment variable."""
        config = {
            "pinecone": {"api_key": "test-key"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.indexing import (
            PineconeMultitenancyIndexingPipeline,
        )

        with (
            patch.object(PineconeMultitenancyIndexingPipeline, "_connect"),
            patch.dict("os.environ", {"PINECONE_INDEX": "env-index"}),
        ):
            pipeline = PineconeMultitenancyIndexingPipeline(config)
            index_name = pipeline._get_index_name()
            assert index_name == "env-index"

    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.load_config")
    def test_close_method(self, mock_load_config, mock_resolve_context):
        """Test close method executes without error."""
        config = {
            "pinecone": {"index": "test-index", "api_key": "test-key"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.indexing import (
            PineconeMultitenancyIndexingPipeline,
        )

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(config)
            # Close should not raise any errors
            pipeline.close()

    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_empty_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with empty documents list."""
        config = {
            "pinecone": {"index": "test-index", "api_key": "test-key"},
            "tenant": {"id": "test_tenant"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.indexing import (
            PineconeMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(config)
            # Run with empty documents list
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 0

    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_custom_tenant_id(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with custom tenant_id parameter."""
        config = {
            "pinecone": {"index": "test-index", "api_key": "test-key"},
            "tenant": {"id": "original_tenant"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="original_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.indexing import (
            PineconeMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": []}
        mock_create_embedder.return_value = mock_embedder

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(config)
            # Run with custom tenant_id
            result = pipeline.run(documents=[], tenant_id="custom_tenant")

            assert result.tenant_id == "custom_tenant"

    def test_create_timing_metrics(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.pinecone.indexing import (
            PineconeMultitenancyIndexingPipeline,
        )

        mock_config = {
            "pinecone": {"index": "test-index", "api_key": "test-key"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.pinecone.indexing.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.pinecone.indexing.TenantContext.resolve",
                return_value=TenantContext("test_tenant"),
            ),
            patch.object(PineconeMultitenancyIndexingPipeline, "_connect"),
        ):
            pipeline = PineconeMultitenancyIndexingPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(5, 100.0)

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test_tenant"


class TestPineconeMultitenancySearch:
    """Extended test suite for Pinecone search pipeline."""

    @patch("vectordb.haystack.multi_tenancy.pinecone.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.load_config")
    def test_get_index_name_from_config(self, mock_load_config, mock_resolve_context):
        """Test _get_index_name returns value from config."""
        config = {
            "pinecone": {"index": "custom-index", "api_key": "test-key"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.search import (
            PineconeMultitenancySearchPipeline,
        )

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(config)
            index_name = pipeline._get_index_name()
            assert index_name == "custom-index"

    @patch("vectordb.haystack.multi_tenancy.pinecone.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.load_config")
    def test_close_method(self, mock_load_config, mock_resolve_context):
        """Test close method executes without error."""
        config = {
            "pinecone": {"index": "test-index", "api_key": "test-key"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.search import (
            PineconeMultitenancySearchPipeline,
        )

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(config)
            # Close should not raise any errors
            pipeline.close()

    @patch("vectordb.haystack.multi_tenancy.pinecone.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.pinecone.search.load_config")
    def test_rag_method_not_enabled(self, mock_load_config, mock_resolve_context):
        """Test rag method raises error when RAG is not enabled."""
        config = {
            "pinecone": {"index": "test-index", "api_key": "test-key"},
            "tenant": {"id": "test_tenant"},
            "rag": {"enabled": False},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.pinecone.search import (
            PineconeMultitenancySearchPipeline,
        )

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(config)
            assert pipeline._rag_pipeline is None

            with pytest.raises(ValueError, match="RAG pipeline not enabled"):
                pipeline.rag("test query")

    def test_create_timing_metrics(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.pinecone.search import (
            PineconeMultitenancySearchPipeline,
        )

        mock_config = {
            "pinecone": {"index": "test-index", "api_key": "test-key"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.pinecone.search.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.pinecone.search.TenantContext.resolve",
                return_value=TenantContext("test_tenant"),
            ),
            patch.object(PineconeMultitenancySearchPipeline, "_connect"),
        ):
            pipeline = PineconeMultitenancySearchPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test_tenant"
