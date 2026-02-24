"""Tests for Weaviate multi-tenancy pipelines.

This module tests Weaviate multi-tenancy pipelines using the new API where
pipelines inherit from BaseMultitenancyPipeline and take config_path: str.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult
from vectordb.haystack.multi_tenancy.weaviate.indexing import (
    WeaviateMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.weaviate.search import (
    WeaviateMultitenancySearchPipeline,
)


@pytest.fixture
def weaviate_config_file(tmp_path: Path) -> Path:
    """Create a temporary Weaviate config file."""
    config_file = tmp_path / "weaviate_config.yaml"
    config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: MultiTenancy
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: test-tenant
"""
    config_file.write_text(config_content)
    return config_file


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

    def test_indexing_pipeline_inherits_from_base(
        self, weaviate_config_file: Path
    ) -> None:
        """Test Weaviate indexing pipeline inherits from BaseMultitenancyPipeline."""
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

            pipeline = WeaviateMultitenancyIndexingPipeline(str(weaviate_config_file))
            assert isinstance(pipeline, BaseMultitenancyPipeline)

    def test_indexing_pipeline_initialization(self, weaviate_config_file: Path) -> None:
        """Test that Weaviate indexing pipeline initializes correctly."""
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

            pipeline = WeaviateMultitenancyIndexingPipeline(str(weaviate_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "weaviate" in pipeline.config

    def test_search_pipeline_inherits_from_base(
        self, weaviate_config_file: Path
    ) -> None:
        """Test that Weaviate search pipeline has expected base attributes.

        Note: WeaviateMultitenancySearchPipeline does not inherit from
        BaseMultitenancyPipeline. It's a standalone pipeline class with
        similar interface.
        """
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

            pipeline = WeaviateMultitenancySearchPipeline(str(weaviate_config_file))
            # Verify it has the expected attributes similar to BaseMultitenancyPipeline
            assert hasattr(pipeline, "config")
            assert hasattr(pipeline, "tenant_context")
            assert pipeline.tenant_context.tenant_id == "test-tenant"

    def test_search_pipeline_initialization(self, weaviate_config_file: Path) -> None:
        """Test that Weaviate search pipeline initializes correctly."""
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

            pipeline = WeaviateMultitenancySearchPipeline(str(weaviate_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "weaviate" in pipeline.config

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Weaviate instance)."""
        pytest.skip("Integration test requires Weaviate service")


class TestWeaviateMultitenancyIndexing:
    """Extended test suite for Weaviate indexing pipeline."""

    def test_get_class_name_from_config(self, tmp_path: Path) -> None:
        """Test _get_class_name returns value from config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: CustomClass
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

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

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            class_name = pipeline._get_class_name()
            assert class_name == "CustomClass"

    def test_get_class_name_default(self, tmp_path: Path) -> None:
        """Test _get_class_name returns default value when not in config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

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

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            class_name = pipeline._get_class_name()
            assert class_name == "MultiTenancy"

    def test_close_method_indexing(self, tmp_path: Path) -> None:
        """Test close method calls client.close()."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

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

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            pipeline.close()
            mock_client.close.assert_called_once()

    def test_run_with_empty_documents(self, tmp_path: Path) -> None:
        """Test run method with empty documents list."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
collection:
  name: TestClass
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 0

    def test_run_with_custom_tenant_id(self, tmp_path: Path) -> None:
        """Test run method with custom tenant_id parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: original-tenant
collection:
  name: TestClass
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": []}
            mock_create_embedder.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[], tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    def test_run_with_documents(self, tmp_path: Path) -> None:
        """Test run method with actual documents."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
collection:
  name: TestClass
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_batch = MagicMock()
            mock_client.batch.return_value = mock_batch
            mock_batch.__enter__ = MagicMock(return_value=mock_batch)
            mock_batch.__exit__ = MagicMock(return_value=False)

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

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=mock_docs)

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 2

    def test_create_timing_metrics_indexing(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

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

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            metrics = pipeline._create_timing_metrics(
                index_operation_ms=100.0,
                total_ms=100.0,
                num_documents=5,
            )

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test-tenant"
            assert metrics.index_operation_ms == 100.0

    def test_connect_with_api_key(self, tmp_path: Path) -> None:
        """Test _connect method with API key authentication."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
  api_key: test-api-key
tenant:
  id: test-tenant
collection:
  name: TestClass
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            WeaviateMultitenancyIndexingPipeline(str(config_file))

            # Verify Client was called with auth_config
            mock_weaviate_client.assert_called_once()
            call_kwargs = mock_weaviate_client.call_args.kwargs
            assert call_kwargs.get("auth_client_secret") is not None

    def test_ensure_tenant_exists_new_tenant(self, tmp_path: Path) -> None:
        """Test _ensure_tenant_exists when tenant doesn't exist."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
collection:
  name: TestClass
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            # Tenant doesn't exist
            mock_client.schema.get_tenant.return_value = [{"name": "other_tenant"}]
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            pipeline._ensure_tenant_exists("TestClass", "new_tenant")

            # Verify add_tenant was called
            mock_client.schema.add_tenant.assert_called()

    def test_ensure_tenant_exists_existing_tenant(self, tmp_path: Path) -> None:
        """Test _ensure_tenant_exists when tenant already exists."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
collection:
  name: TestClass
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client

            # Tenant already exists
            mock_client.schema.get_tenant.return_value = [{"name": "test_tenant"}]
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
            pipeline._ensure_tenant_exists("TestClass", "test_tenant")

            # Verify add_tenant was NOT called
            mock_client.schema.add_tenant.assert_not_called()

    def test_connect_with_env_vars(self, tmp_path: Path) -> None:
        """Test _connect method uses environment variables when URL not in config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
tenant:
  id: test-tenant
collection:
  name: TestClass
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch.dict("os.environ", {"WEAVIATE_URL": "http://env-host:8080"}),
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            WeaviateMultitenancyIndexingPipeline(str(config_file))

            # Verify Client was called with URL from env
            mock_weaviate_client.assert_called_once()
            call_kwargs = mock_weaviate_client.call_args.kwargs
            assert call_kwargs.get("url") == "http://env-host:8080"


class TestWeaviateMultitenancySearch:
    """Extended test suite for Weaviate search pipeline."""

    def test_query_method(self, tmp_path: Path) -> None:
        """Test query method returns results."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
collection:
  name: TestClass
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.weaviate.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("weaviate.Client") as mock_weaviate_client,
        ):
            mock_client = MagicMock()
            mock_weaviate_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

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

            pipeline = WeaviateMultitenancySearchPipeline(str(config_file))
            result = pipeline.query("test query")

            assert result.tenant_id == "test-tenant"
            assert result.query == "test query"

    def test_close_method_search(self, tmp_path: Path) -> None:
        """Test close method calls client.close()."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
collection:
  name: TestClass
"""
        config_file.write_text(config_content)

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

            pipeline = WeaviateMultitenancySearchPipeline(str(config_file))
            pipeline.close()
            mock_client.close.assert_called_once()

    def test_create_timing_metrics_search(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

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

            pipeline = WeaviateMultitenancySearchPipeline(str(config_file))
            # Search pipeline's _create_timing_metrics takes just total_ms
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test-tenant"
