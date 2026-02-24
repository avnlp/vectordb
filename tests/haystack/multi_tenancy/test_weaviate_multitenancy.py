"""Tests for Weaviate multi-tenancy pipelines."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


class TestWeaviateMultitenancy:
    """Test suite for Weaviate multi-tenancy pipelines."""

    def test_tenant_context_resolution(self) -> None:
        """Test tenant context resolution."""
        tenant_ctx = TenantContext(tenant_id="test_tenant")
        assert tenant_ctx.tenant_id == "test_tenant"

    def test_tenant_context_from_config(self) -> None:
        """Test tenant context resolution from config."""
        config = {"tenant": {"id": "config_tenant"}}
        tenant_ctx = TenantContext.resolve(None, config)
        assert tenant_ctx.tenant_id == "config_tenant"

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test that Weaviate indexing pipeline initializes correctly."""
        weaviate_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = weaviate_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(weaviate_config)
            assert pipeline.config == weaviate_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @patch("vectordb.haystack.multi_tenancy.weaviate.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test that Weaviate search pipeline initializes correctly."""
        weaviate_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = weaviate_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.search import (
            WeaviateMultitenancySearchPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancySearchPipeline(weaviate_config)
            assert pipeline.config == weaviate_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Weaviate instance)."""
        pytest.skip("Integration test requires Weaviate service")


class TestWeaviateMultitenancyIndexing:
    """Extended test suite for Weaviate indexing pipeline."""

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    def test_get_class_name_from_config(self, mock_load_config, mock_resolve_context):
        """Test _get_class_name returns value from config."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "collection": {"name": "CustomClass"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            class_name = pipeline._get_class_name()
            assert class_name == "CustomClass"

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    def test_get_class_name_default(self, mock_load_config, mock_resolve_context):
        """Test _get_class_name returns default value when not in config."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            class_name = pipeline._get_class_name()
            assert class_name == "MultiTenancy"

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    def test_close_method_indexing(self, mock_load_config, mock_resolve_context):
        """Test close method calls client.close()."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            pipeline.close()
            mock_client.close.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_empty_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with empty documents list."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("weaviate.Client") as mock_weaviate_client:
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 0

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_custom_tenant_id(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with custom tenant_id parameter."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "original_tenant"},
            "collection": {"name": "TestClass"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="original_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": []}
        mock_create_embedder.return_value = mock_embedder

        with patch("weaviate.Client") as mock_weaviate_client:
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[], tenant_id="custom_tenant")

            assert result.tenant_id == "custom_tenant"

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with actual documents."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        # Create mock documents with embeddings
        mock_docs = [
            Document(
                content="Test document 1",
                meta={"source": "test"},
                embedding=[0.1] * 1024,
            ),
            Document(
                content="Test document 2",
                meta={"source": "test"},
                embedding=[0.2] * 1024,
            ),
        ]
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": mock_docs}
        mock_create_embedder.return_value = mock_embedder

        with patch("weaviate.Client") as mock_weaviate_client:
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_batch = MagicMock()
            mock_client.batch.return_value = mock_batch
            mock_batch.__enter__ = MagicMock(return_value=mock_batch)
            mock_batch.__exit__ = MagicMock(return_value=False)

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=mock_docs)

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 2

    def test_create_timing_metrics_indexing(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        mock_config = {
            "weaviate": {"url": "http://localhost:8080"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve",
                return_value=TenantContext("test_tenant"),
            ),
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(5, 100.0)

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test_tenant"
            assert metrics.index_operation_ms == 100.0

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_connect_with_api_key(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _connect method with API key authentication."""
        config = {
            "weaviate": {"url": "http://localhost:8080", "api_key": "test-api-key"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("weaviate.Client") as mock_weaviate_client:
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            WeaviateMultitenancyIndexingPipeline(config)

            # Verify Client was called with auth_config
            mock_weaviate_client.assert_called_once()
            call_kwargs = mock_weaviate_client.call_args.kwargs
            assert call_kwargs.get("auth_client_secret") is not None

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_ensure_tenant_exists_new_tenant(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _ensure_tenant_exists when tenant doesn't exist."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("weaviate.Client") as mock_weaviate_client:
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            # Tenant doesn't exist
            mock_client.schema.get_tenant.return_value = [{"name": "other_tenant"}]

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            pipeline._ensure_tenant_exists("TestClass", "new_tenant")

            # Verify add_tenant was called
            mock_client.schema.add_tenant.assert_called()

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_ensure_tenant_exists_existing_tenant(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _ensure_tenant_exists when tenant already exists."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("weaviate.Client") as mock_weaviate_client:
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            # Tenant already exists
            mock_client.schema.get_tenant.return_value = [{"name": "test_tenant"}]

            pipeline = WeaviateMultitenancyIndexingPipeline(config)
            pipeline._ensure_tenant_exists("TestClass", "test_tenant")

            # Verify add_tenant was NOT called
            mock_client.schema.add_tenant.assert_not_called()

    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_connect_with_env_vars(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _connect method uses environment variables when URL not in config."""
        config = {
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.indexing import (
            WeaviateMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with (
            patch.dict("os.environ", {"WEAVIATE_URL": "http://env-host:8080"}),
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            WeaviateMultitenancyIndexingPipeline(config)

            # Verify Client was called with URL from env
            mock_weaviate_client.assert_called_once()
            call_kwargs = mock_weaviate_client.call_args.kwargs
            assert call_kwargs.get("url") == "http://env-host:8080"


class TestWeaviateMultitenancySearch:
    """Extended test suite for Weaviate search pipeline."""

    @patch("vectordb.haystack.multi_tenancy.weaviate.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder")
    def test_query_method(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method returns results."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.search import (
            WeaviateMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch("weaviate.Client") as mock_weaviate_client:
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            # Mock search result
            mock_result = MagicMock()
            mock_result.objects = [
                MagicMock(
                    uuid="test-uuid",
                    properties={"content": "Test content", "tenant_id": "test_tenant"},
                    vector=[0.1] * 1024,
                ),
            ]
            mock_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.do.return_value = mock_result

            pipeline = WeaviateMultitenancySearchPipeline(config)
            result = pipeline.query("test query")

            assert result.tenant_id == "test_tenant"
            assert result.query == "test query"

    @patch("vectordb.haystack.multi_tenancy.weaviate.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.weaviate.search.load_config")
    def test_close_method_search(self, mock_load_config, mock_resolve_context):
        """Test close method calls client.close()."""
        config = {
            "weaviate": {"url": "http://localhost:8080"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "TestClass"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.weaviate.search import (
            WeaviateMultitenancySearchPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancySearchPipeline(config)
            pipeline.close()
            mock_client.close.assert_called_once()

    def test_create_timing_metrics_search(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.weaviate.search import (
            WeaviateMultitenancySearchPipeline,
        )

        mock_config = {
            "weaviate": {"url": "http://localhost:8080"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.search.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.search.TenantContext.resolve",
                return_value=TenantContext("test_tenant"),
            ),
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = WeaviateMultitenancySearchPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test_tenant"
