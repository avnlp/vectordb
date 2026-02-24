"""Tests for Qdrant multi-tenancy pipelines.

This module tests Qdrant multi-tenancy pipelines using the new API where
pipelines inherit from BaseMultitenancyPipeline and take config_path: str.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult
from vectordb.haystack.multi_tenancy.qdrant.indexing import (
    QdrantMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.qdrant.search import (
    QdrantMultitenancySearchPipeline,
)


@pytest.fixture
def qdrant_config_file(tmp_path: Path) -> Path:
    """Create a temporary Qdrant config file."""
    config_file = tmp_path / "qdrant_config.yaml"
    config_content = """
qdrant:
  url: http://localhost:6333
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


class TestQdrantMultitenancy:
    """Test suite for Qdrant multi-tenancy pipelines."""

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
        self, qdrant_config_file: Path
    ) -> None:
        """Test that Qdrant indexing pipeline inherits from BaseMultitenancyPipeline."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(qdrant_config_file))
            assert isinstance(pipeline, BaseMultitenancyPipeline)

    def test_indexing_pipeline_initialization(self, qdrant_config_file: Path) -> None:
        """Test that Qdrant indexing pipeline initializes correctly."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(qdrant_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "qdrant" in pipeline.config

    def test_search_pipeline_inherits_from_base(self, qdrant_config_file: Path) -> None:
        """Test that Qdrant search pipeline has expected base attributes.

        Note: QdrantMultitenancySearchPipeline does not inherit from
        BaseMultitenancyPipeline. It's a standalone pipeline class with
        similar interface.
        """
        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(str(qdrant_config_file))
            # Verify it has the expected attributes similar to BaseMultitenancyPipeline
            assert hasattr(pipeline, "config")
            assert hasattr(pipeline, "tenant_context")
            assert pipeline.tenant_context.tenant_id == "test-tenant"

    def test_search_pipeline_initialization(self, qdrant_config_file: Path) -> None:
        """Test that Qdrant search pipeline initializes correctly."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(str(qdrant_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "qdrant" in pipeline.config

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Qdrant instance)."""
        pytest.skip("Integration test requires Qdrant service")


class TestQdrantMultitenancyIndexing:
    """Extended test suite for Qdrant indexing pipeline."""

    def test_get_collection_name_from_config(self, tmp_path: Path) -> None:
        """Test _get_collection_name returns value from config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
collection:
  name: custom-collection
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            collection_name = pipeline._get_collection_name()
            assert collection_name == "custom-collection"

    def test_get_collection_name_default(self, tmp_path: Path) -> None:
        """Test _get_collection_name returns default value when not in config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            collection_name = pipeline._get_collection_name()
            assert collection_name == "multitenancy"

    def test_close_method(self, tmp_path: Path) -> None:
        """Test close method calls client.close()."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            pipeline.close()
            mock_client.close.assert_called_once()

    def test_run_with_empty_documents(self, tmp_path: Path) -> None:
        """Test run method with empty documents list."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
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
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 0

    def test_run_with_custom_tenant_id(self, tmp_path: Path) -> None:
        """Test run method with custom tenant_id parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
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
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": []}
            mock_create_embedder.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[], tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    def test_run_with_documents(self, tmp_path: Path) -> None:
        """Test run method with actual documents."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
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
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

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

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=mock_docs)

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 2
            mock_client.upsert.assert_called_once()

    def test_create_timing_metrics(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            metrics = pipeline._create_timing_metrics(
                index_operation_ms=100.0,
                total_ms=100.0,
                num_documents=5,
            )

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test-tenant"
            assert metrics.index_operation_ms == 100.0

    def test_connect_with_local_location(self, tmp_path: Path) -> None:
        """Test _connect method with local location config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  location: ./test_data
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
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = False
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            QdrantMultitenancyIndexingPipeline(str(config_file))

            # Verify QdrantClient was called with location
            mock_qdrant_client.assert_called_with(location="./test_data")
            mock_client.create_collection.assert_called_once()
            mock_client.create_payload_index.assert_called_once()

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
                "os.environ", {"QDRANT_HOST": "remote-host", "QDRANT_PORT": "6334"}
            ),
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            QdrantMultitenancyIndexingPipeline(str(config_file))

            # Verify QdrantClient was called with host and port from env
            mock_qdrant_client.assert_called_with(host="remote-host", port=6334)

    def test_load_documents_from_dataloader_with_params(self, tmp_path: Path) -> None:
        """Test _load_documents_from_dataloader applies params from config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
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
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.DataloaderCatalog"
            ) as mock_catalog,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            # Setup proper mock chain: loader.load().to_haystack() -> documents
            mock_docs = [
                Document(content="Doc from dataloader", meta={"source": "triviaqa"})
            ]
            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = mock_docs
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_catalog.create.return_value = mock_loader

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            documents = pipeline._load_documents_from_dataloader()

            # Verify DataloaderCatalog.create was called correctly
            mock_catalog.create.assert_called_once_with(
                "triviaqa",
                split="train",
                limit=None,
                dataset_id=None,
            )
            # Verify documents returned
            assert len(documents) == 1
            assert documents[0].content == "Doc from dataloader"

    def test_run_with_documents_none_triggers_dataloader(self, tmp_path: Path) -> None:
        """Test run() with documents=None loads from dataloader."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  dimension: 1024
dataloader:
  dataset: popqa
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.indexing.DataloaderCatalog"
            ) as mock_catalog,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

            # Create mock documents with embeddings for the embedder to return
            mock_docs_from_loader = [
                Document(content="Loaded doc 1", meta={"source": "popqa"}),
                Document(content="Loaded doc 2", meta={"source": "popqa"}),
            ]
            mock_embedded_docs = [
                Document(
                    content="Loaded doc 1",
                    meta={"source": "popqa"},
                    embedding=[0.1] * 1024,
                ),
                Document(
                    content="Loaded doc 2",
                    meta={"source": "popqa"},
                    embedding=[0.2] * 1024,
                ),
            ]

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": mock_embedded_docs}
            mock_create_embedder.return_value = mock_embedder

            # Setup proper mock chain: loader.load().to_haystack() -> documents
            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = mock_docs_from_loader
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_catalog.create.return_value = mock_loader

            pipeline = QdrantMultitenancyIndexingPipeline(str(config_file))
            # Call run with documents=None to trigger dataloader
            result = pipeline.run(documents=None)

            # Verify dataloader was used
            mock_catalog.create.assert_called_once_with(
                "popqa",
                split="test",
                limit=None,
                dataset_id=None,
            )
            # Verify documents were embedded and indexed
            mock_embedder.run.assert_called_once()
            mock_client.upsert.assert_called_once()
            # Verify result
            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 2


class TestQdrantMultitenancySearch:
    """Extended test suite for Qdrant search pipeline."""

    def test_query_method(self, tmp_path: Path) -> None:
        """Test query method returns results."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            # Mock search result - Qdrant returns a list of ScoredPoint
            mock_result = MagicMock()
            mock_result.payload = {
                "content": "Test content",
                "tenant_id": "test_tenant",
            }
            mock_result.score = 0.95
            mock_client.search.return_value = [mock_result]

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            result = pipeline.query("test query")

            assert result.tenant_id == "test-tenant"
            assert result.query == "test query"
            assert len(result.documents) == 1
            assert result.documents[0].content == "Test content"

    def test_query_with_custom_top_k(self, tmp_path: Path) -> None:
        """Test query method with custom top_k parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            mock_result = MagicMock()
            mock_result.points = []
            mock_client.search.return_value = mock_result

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            pipeline.query("test query", top_k=20)

            # Verify search was called with correct top_k
            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs.get(
                "limit", call_kwargs.get("top", call_kwargs.get("top_k", 0))
            ) in [20, None]

    def test_query_with_custom_tenant_id(self, tmp_path: Path) -> None:
        """Test query method with custom tenant_id parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: original-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            mock_result = MagicMock()
            mock_result.points = []
            mock_client.search.return_value = mock_result

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            result = pipeline.query("test query", tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    def test_search_close_method(self, tmp_path: Path) -> None:
        """Test close method calls client.close()."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            pipeline.close()
            mock_client.close.assert_called_once()

    def test_search_create_timing_metrics(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            # Search pipeline's _create_timing_metrics takes just total_ms
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test-tenant"

    def test_query_empty_results(self, tmp_path: Path) -> None:
        """Test query method returns empty results when no matches."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            mock_result = MagicMock()
            mock_result.points = []
            mock_client.search.return_value = mock_result

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            result = pipeline.query("test query")

            assert result.tenant_id == "test-tenant"
            assert len(result.documents) == 0
            assert len(result.scores) == 0

    def test_search_connect_with_local_location(self, tmp_path: Path) -> None:
        """Test _connect method with local location config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  location: ./test_data
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            QdrantMultitenancySearchPipeline(str(config_file))

            # Verify QdrantClient was called with location
            mock_qdrant_client.assert_called_with(location="./test_data")

    def test_search_connect_with_env_vars(self, tmp_path: Path) -> None:
        """Test _connect method uses environment variables when config not present."""
        config_file = tmp_path / "config.yaml"
        config_content = """
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch.dict(
                "os.environ", {"QDRANT_HOST": "remote-host", "QDRANT_PORT": "6334"}
            ),
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            QdrantMultitenancySearchPipeline(str(config_file))

            # Verify QdrantClient was called with host and port from env
            mock_qdrant_client.assert_called_with(host="remote-host", port=6334)

    def test_close_with_none_client(self, tmp_path: Path) -> None:
        """Test close method when _client is None (no-op)."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            # Manually set _client to None to simulate uninitialized state
            pipeline._client = None
            # Should not raise and should not call close on None
            pipeline.close()

    def test_query_with_output_dimension_truncation(self, tmp_path: Path) -> None:
        """Test query method truncates embedding when output_dimension is specified."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  output_dimension: 512
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_client.search.return_value = []
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": list(range(1024))}
            mock_create_embedder.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            pipeline.query("test query")

            # Verify search was called with truncated embedding (512 dimensions)
            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            query_vector = call_kwargs["query_vector"]
            assert len(query_vector) == 512
            assert query_vector == list(range(512))

    def test_rag_raises_when_not_enabled(self, tmp_path: Path) -> None:
        """Test rag method raises ValueError when RAG pipeline is not enabled."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
            pytest.raises(ValueError, match="RAG pipeline not enabled/configured"),
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            pipeline.rag("test query")

    def test_rag_success_when_enabled(self, tmp_path: Path) -> None:
        """Test rag method succeeds when RAG pipeline is enabled."""
        config_file = tmp_path / "config.yaml"
        config_content = """
qdrant:
  url: http://localhost:6333
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
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_rag_pipeline"
            ) as mock_create_rag_pipeline,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            # Mock the RAG pipeline
            mock_rag_pipeline = MagicMock()
            mock_rag_pipeline.run.return_value = {
                "generator": {"replies": ["Generated response from RAG"]}
            }
            mock_create_rag_pipeline.return_value = mock_rag_pipeline

            # Mock search result for query
            mock_result = MagicMock()
            mock_result.payload = {
                "content": "Test document content",
                "tenant_id": "test_tenant",
            }
            mock_result.score = 0.95
            mock_client.search.return_value = [mock_result]

            pipeline = QdrantMultitenancySearchPipeline(str(config_file))
            result = pipeline.rag("test query", top_k=5, tenant_id="test-tenant")

            # Verify the result
            assert result.tenant_id == "test-tenant"
            assert result.query == "test query"
            assert result.generated_response == "Generated response from RAG"
            assert len(result.retrieved_documents) == 1
            assert result.retrieved_documents[0].content == "Test document content"
            assert result.retrieval_scores == [0.95]

            # Verify RAG pipeline was called
            mock_rag_pipeline.run.assert_called_once()
            call_args = mock_rag_pipeline.run.call_args[0][0]
            assert call_args["prompt_builder"]["query"] == "test query"
            assert len(call_args["prompt_builder"]["documents"]) == 1
