"""Tests for Milvus multi-tenancy pipelines.

This module tests Milvus multi-tenancy pipelines using the new API where
pipelines inherit from BaseMultitenancyPipeline and take config_path: str.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult
from vectordb.haystack.multi_tenancy.milvus.indexing import (
    MilvusMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.milvus.search import (
    MilvusMultitenancySearchPipeline,
)


@pytest.fixture
def milvus_config_file(tmp_path: Path) -> Path:
    """Create a temporary Milvus config file."""
    config_file = tmp_path / "milvus_config.yaml"
    config_content = """
milvus:
  uri: http://localhost:19530
collection:
  name: test_collection
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: test-tenant
"""
    config_file.write_text(config_content)
    return config_file


class TestMilvusMultitenancy:
    """Test suite for Milvus multi-tenancy pipelines."""

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
        self, milvus_config_file: Path
    ) -> None:
        """Test that Milvus indexing pipeline inherits from BaseMultitenancyPipeline."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(milvus_config_file))
            assert isinstance(pipeline, BaseMultitenancyPipeline)

    def test_indexing_pipeline_initialization(self, milvus_config_file: Path) -> None:
        """Test that Milvus indexing pipeline initializes correctly."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(milvus_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "milvus" in pipeline.config

    def test_search_pipeline_inherits_from_base(self, milvus_config_file: Path) -> None:
        """Test that Milvus search pipeline has expected base attributes.

        Note: MilvusMultitenancySearchPipeline does not inherit from
        BaseMultitenancyPipeline. It's a standalone pipeline class with
        similar interface.
        """
        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(milvus_config_file))
            # Verify it has the expected attributes similar to BaseMultitenancyPipeline
            assert hasattr(pipeline, "config")
            assert hasattr(pipeline, "tenant_context")
            assert pipeline.tenant_context.tenant_id == "test-tenant"

    def test_search_pipeline_initialization(self, milvus_config_file: Path) -> None:
        """Test that Milvus search pipeline initializes correctly."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(milvus_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "milvus" in pipeline.config

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Milvus instance)."""
        pytest.skip("Integration test requires Milvus service")


class TestMilvusMultitenancyIndexing:
    """Extended test suite for Milvus indexing pipeline."""

    def test_get_collection_name_from_config(self, tmp_path: Path) -> None:
        """Test _get_collection_name returns value from config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
collection:
  name: custom-collection
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            collection_name = pipeline._get_collection_name()
            assert collection_name == "custom-collection"

    def test_get_collection_name_default(self, tmp_path: Path) -> None:
        """Test _get_collection_name returns default value when not in config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            collection_name = pipeline._get_collection_name()
            assert collection_name == "multitenancy"

    def test_close_method(self, tmp_path: Path) -> None:
        """Test close method calls db.close()."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            pipeline.close()
            mock_db.close.assert_called_once()

    def test_run_with_empty_documents(self, tmp_path: Path) -> None:
        """Test run method with empty documents list."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 0

    def test_run_with_custom_tenant_id(self, tmp_path: Path) -> None:
        """Test run method with custom tenant_id parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: original-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": []}
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[], tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    def test_run_with_documents(self, tmp_path: Path) -> None:
        """Test run method with actual documents."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

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

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=mock_docs)

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 2
            mock_db.insert_documents.assert_called_once()

    def test_create_timing_metrics(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            metrics = pipeline._create_timing_metrics(
                index_operation_ms=100.0,
                total_ms=100.0,
                num_documents=5,
            )

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test-tenant"
            assert metrics.index_operation_ms == 100.0

    def test_connect_with_env_vars(self, tmp_path: Path) -> None:
        """Test _connect method uses environment variables when config not present."""
        config_file = tmp_path / "config.yaml"
        config_content = """
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch.dict(
                "os.environ",
                {"MILVUS_URI": "http://env-host:19530", "MILVUS_TOKEN": "test-token"},
            ),
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            MilvusMultitenancyIndexingPipeline(str(config_file))

            # Verify MilvusVectorDB was called with URI from env
            mock_milvus_db.assert_called_once()
            call_kwargs = mock_milvus_db.call_args.kwargs
            assert call_kwargs.get("uri") == "http://env-host:19530"
            assert call_kwargs.get("token") == "test-token"

    def test_tenant_field_constant(self, tmp_path: Path) -> None:
        """Test that TENANT_FIELD constant is correctly defined."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            assert pipeline.TENANT_FIELD == "tenant_id"

    def test_load_documents_from_dataloader_with_params(self, tmp_path: Path) -> None:
        """Test _load_documents_from_dataloader sets attributes from params config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
dataloader:
  dataset: triviaqa
  params:
    split: train
    max_samples: 100
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.DataloaderCatalog"
            ) as mock_catalog,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            # Setup proper mock chain: loader.load().to_haystack() -> documents
            mock_docs = [
                Document(content="Doc 1", meta={}),
                Document(content="Doc 2", meta={}),
            ]
            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = mock_docs
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_catalog.create.return_value = mock_loader

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            result = pipeline._load_documents_from_dataloader()

            # Verify DataloaderCatalog.create was called with correct params from config
            mock_catalog.create.assert_called_once_with(
                "triviaqa",
                split="train",
                limit=None,
                dataset_id=None,
            )
            mock_loader.load.assert_called_once()
            mock_dataset.to_haystack.assert_called_once()
            assert result == mock_docs

    def test_run_with_documents_none_uses_dataloader(self, tmp_path: Path) -> None:
        """Test run() with documents=None loads from dataloader."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
dataloader:
  dataset: popqa
  params:
    limit: 50
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.DataloaderCatalog"
            ) as mock_catalog,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            # Setup proper mock chain: loader.load().to_haystack() -> documents
            mock_docs = [
                Document(content="Loaded doc 1", meta={}, embedding=[0.1] * 1024),
                Document(content="Loaded doc 2", meta={}, embedding=[0.2] * 1024),
            ]

            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = mock_docs
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_catalog.create.return_value = mock_loader

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": mock_docs}
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=None)

            # Verify DataloaderCatalog.create was called with correct params from config
            mock_catalog.create.assert_called_once_with(
                "popqa",
                split="test",
                limit=50,
                dataset_id=None,
            )
            mock_loader.load.assert_called_once()
            mock_dataset.to_haystack.assert_called_once()
            assert isinstance(result, TenantIndexResult)
            assert result.documents_indexed == 2
            mock_db.insert_documents.assert_called_once()


class TestMilvusMultitenancySearch:
    """Extended test suite for Milvus search pipeline."""

    def test_connect_warms_embedder_and_sets_rag_pipeline(self, tmp_path: Path) -> None:
        """Test connect initializes embedder and RAG pipeline when enabled."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
rag:
  enabled: true
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_rag_pipeline"
            ) as mock_create_rag_pipeline,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder
            mock_rag_pipeline = MagicMock()
            mock_create_rag_pipeline.return_value = mock_rag_pipeline

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))

            mock_embedder.warm_up.assert_called_once()
            mock_create_rag_pipeline.assert_called_once()
            assert pipeline._rag_pipeline == mock_rag_pipeline

    def test_query_method(self, tmp_path: Path) -> None:
        """Test query method returns results."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            # Mock search result
            mock_db.retrieve.return_value = [
                Document(
                    content="Test content",
                    meta={"tenant_id": "test_tenant"},
                    embedding=[0.1] * 1024,
                )
            ]

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            result = pipeline.query("test query")

            assert result.tenant_id == "test-tenant"
            assert result.query == "test query"

    def test_query_truncates_embedding_when_output_dimension_set(
        self, tmp_path: Path
    ) -> None:
        """Test query truncates embeddings when output dimension is configured."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  output_dimension: 4
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_db.retrieve.return_value = []
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            pipeline.query("test query")

            call_kwargs = mock_db.retrieve.call_args.kwargs
            assert call_kwargs["query_embedding"] == [0.1, 0.2, 0.3, 0.4]

    def test_query_with_custom_top_k(self, tmp_path: Path) -> None:
        """Test query method with custom top_k parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_db.retrieve.return_value = []
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            pipeline.query("test query", top_k=20)

            # Verify retrieve was called
            mock_db.retrieve.assert_called_once()

    def test_query_with_custom_tenant_id(self, tmp_path: Path) -> None:
        """Test query method with custom tenant_id parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: original-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_db.retrieve.return_value = []
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            result = pipeline.query("test query", tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    def test_close_method(self, tmp_path: Path) -> None:
        """Test close method calls db.close()."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            pipeline.close()
            mock_db.close.assert_called_once()

    def test_close_no_db_connection(self, tmp_path: Path) -> None:
        """Test close handles missing db connection gracefully."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            pipeline._db = None
            pipeline.close()

    def test_create_timing_metrics(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            # Search pipeline's _create_timing_metrics takes just total_ms
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test-tenant"

    def test_rag_disabled_raises_error(self, tmp_path: Path) -> None:
        """Test rag raises when pipeline is not configured/enabled."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
rag:
  enabled: false
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
            pytest.raises(ValueError, match="RAG pipeline not enabled/configured"),
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            pipeline.rag("test query")

    def test_rag_returns_generated_response(self, tmp_path: Path) -> None:
        """Test rag returns generated response with retrieved documents."""
        config_file = tmp_path / "config.yaml"
        config_content = """
milvus:
  uri: http://localhost:19530
tenant:
  id: test-tenant
collection:
  name: test_collection
rag:
  enabled: true
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.create_rag_pipeline"
            ) as mock_create_rag_pipeline,
        ):
            mock_db = MagicMock()
            mock_db.retrieve.return_value = [
                Document(content="Result", meta={"tenant_id": "test_tenant"})
            ]
            mock_milvus_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 8}
            mock_create_embedder.return_value = mock_embedder
            mock_rag_pipeline = MagicMock()
            mock_rag_pipeline.run.return_value = {
                "generator": {"replies": ["Generated answer"]}
            }
            mock_create_rag_pipeline.return_value = mock_rag_pipeline

            pipeline = MilvusMultitenancySearchPipeline(str(config_file))
            result = pipeline.rag("test query", top_k=3)

            assert result.generated_response == "Generated answer"
            assert len(result.retrieved_documents) == 1
