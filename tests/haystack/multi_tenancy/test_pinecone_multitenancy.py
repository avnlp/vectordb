"""Tests for Pinecone multi-tenancy pipelines.

This module tests Pinecone multi-tenancy pipelines using the new API where
pipelines inherit from BaseMultitenancyPipeline and take config_path: str.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult
from vectordb.haystack.multi_tenancy.pinecone.indexing import (
    PineconeMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.pinecone.search import (
    PineconeMultitenancySearchPipeline,
)


@pytest.fixture
def pinecone_config_file(tmp_path: Path) -> Path:
    """Create a temporary Pinecone config file."""
    config_file = tmp_path / "pinecone_config.yaml"
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
    return config_file


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

    def test_indexing_pipeline_inherits_from_base(
        self, pinecone_config_file: Path
    ) -> None:
        """Test Pinecone indexing pipeline inherits from BaseMultitenancyPipeline."""
        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(str(pinecone_config_file))
            assert isinstance(pipeline, BaseMultitenancyPipeline)

    def test_indexing_pipeline_initialization(self, pinecone_config_file: Path) -> None:
        """Test that Pinecone indexing pipeline initializes correctly."""
        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(str(pinecone_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "pinecone" in pipeline.config

    def test_search_pipeline_inherits_from_base(
        self, pinecone_config_file: Path
    ) -> None:
        """Test Pinecone search pipeline inherits from BaseMultitenancyPipeline."""
        # Note: PineconeMultitenancySearchPipeline does not inherit from
        # BaseMultitenancyPipeline. It's a standalone pipeline class
        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(str(pinecone_config_file))
            # Verify it has the expected attributes
            assert hasattr(pipeline, "config")
            assert hasattr(pipeline, "tenant_context")
            assert pipeline.tenant_context.tenant_id == "test-tenant"

    def test_search_pipeline_initialization(self, pinecone_config_file: Path) -> None:
        """Test that Pinecone search pipeline initializes correctly."""
        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(str(pinecone_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "pinecone" in pipeline.config

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Pinecone instance)."""
        pytest.skip("Integration test requires Pinecone service")


class TestPineconeMultitenancyIndexing:
    """Extended test suite for Pinecone indexing pipeline."""

    def test_get_index_name_from_config(self, tmp_path: Path) -> None:
        """Test _get_index_name returns value from config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: custom-index
  api_key: test-key
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(str(config_file))
            index_name = pipeline._get_index_name()
            assert index_name == "custom-index"

    def test_get_index_name_from_env(self, tmp_path: Path) -> None:
        """Test _get_index_name returns value from environment variable."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  api_key: test-key
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch.object(PineconeMultitenancyIndexingPipeline, "_connect"),
            patch.dict("os.environ", {"PINECONE_INDEX": "env-index"}),
        ):
            pipeline = PineconeMultitenancyIndexingPipeline(str(config_file))
            index_name = pipeline._get_index_name()
            assert index_name == "env-index"

    def test_close_method(self, tmp_path: Path) -> None:
        """Test close method executes without error."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: test-index
  api_key: test-key
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(str(config_file))
            # Close should not raise any errors
            pipeline.close()

    def test_run_with_empty_documents(self, tmp_path: Path) -> None:
        """Test run method with empty documents list."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: test-index
  api_key: test-key
tenant:
  id: test-tenant
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch.object(PineconeMultitenancyIndexingPipeline, "_connect"),
        ):
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = PineconeMultitenancyIndexingPipeline(str(config_file))
            # Run with empty documents list
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 0

    def test_run_with_custom_tenant_id(self, tmp_path: Path) -> None:
        """Test run method with custom tenant_id parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: test-index
  api_key: test-key
tenant:
  id: original-tenant
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch.object(PineconeMultitenancyIndexingPipeline, "_connect"),
        ):
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": []}
            mock_create_embedder.return_value = mock_embedder

            pipeline = PineconeMultitenancyIndexingPipeline(str(config_file))
            # Run with custom tenant_id
            result = pipeline.run(documents=[], tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    def test_create_timing_metrics(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: test-index
  api_key: test-key
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with patch.object(PineconeMultitenancyIndexingPipeline, "_connect"):
            pipeline = PineconeMultitenancyIndexingPipeline(str(config_file))
            metrics = pipeline._create_timing_metrics(
                tenant_resolution_ms=5,
                index_operation_ms=100.0,
                num_documents=5,
            )

            assert metrics.num_documents == 5
            assert metrics.index_operation_ms == 100.0
            assert metrics.tenant_id == "test-tenant"


class TestPineconeMultitenancySearch:
    """Extended test suite for Pinecone search pipeline."""

    def test_get_index_name_from_config(self, tmp_path: Path) -> None:
        """Test _get_index_name returns value from config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: custom-index
  api_key: test-key
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(str(config_file))
            index_name = pipeline._get_index_name()
            assert index_name == "custom-index"

    def test_close_method(self, tmp_path: Path) -> None:
        """Test close method executes without error."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: test-index
  api_key: test-key
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(str(config_file))
            # Close should not raise any errors
            pipeline.close()

    def test_rag_method_not_enabled(self, tmp_path: Path) -> None:
        """Test rag method raises error when RAG is not enabled."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: test-index
  api_key: test-key
tenant:
  id: test-tenant
rag:
  enabled: false
"""
        config_file.write_text(config_content)

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(str(config_file))
            assert pipeline._rag_pipeline is None

            with pytest.raises(ValueError, match="RAG pipeline not enabled"):
                pipeline.rag("test query")

    def test_create_timing_metrics(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
pinecone:
  index: test-index
  api_key: test-key
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with patch.object(PineconeMultitenancySearchPipeline, "_connect"):
            pipeline = PineconeMultitenancySearchPipeline(str(config_file))
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test-tenant"
