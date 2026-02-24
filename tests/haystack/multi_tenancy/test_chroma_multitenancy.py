"""Tests for Chroma multi-tenancy pipelines.

This module tests Chroma multi-tenancy pipelines using the new API where
pipelines inherit from BaseMultitenancyPipeline and take config_path: str.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.chroma.indexing import (
    ChromaMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.chroma.search import (
    ChromaMultitenancySearchPipeline,
)
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


@pytest.fixture
def chroma_config_file(tmp_path: Path) -> Path:
    """Create a temporary Chroma config file."""
    config_file = tmp_path / "chroma_config.yaml"
    config_content = """
chroma:
  persist_dir: ./test_db
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


class TestChromaMultitenancy:
    """Test suite for Chroma multi-tenancy pipelines."""

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
        self, chroma_config_file: Path
    ) -> None:
        """Test that Chroma indexing pipeline inherits from BaseMultitenancyPipeline."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"),
        ):
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(chroma_config_file))
            assert isinstance(pipeline, BaseMultitenancyPipeline)

    def test_indexing_pipeline_initialization(self, chroma_config_file: Path) -> None:
        """Test that Chroma indexing pipeline initializes correctly."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"),
        ):
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(chroma_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "chroma" in pipeline.config

    def test_search_pipeline_inherits_from_base(self, chroma_config_file: Path) -> None:
        """Test that Chroma search pipeline has expected base attributes.

        Note: ChromaMultitenancySearchPipeline does not inherit from
        BaseMultitenancyPipeline. It's a standalone pipeline class with
        similar interface.
        """
        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"),
        ):
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(str(chroma_config_file))
            # Verify it has the expected attributes similar to BaseMultitenancyPipeline
            assert hasattr(pipeline, "config")
            assert hasattr(pipeline, "tenant_context")
            assert pipeline.tenant_context.tenant_id == "test-tenant"

    def test_search_pipeline_initialization(self, chroma_config_file: Path) -> None:
        """Test that Chroma search pipeline initializes correctly."""
        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"),
        ):
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(str(chroma_config_file))
            assert pipeline.tenant_context.tenant_id == "test-tenant"
            assert "chroma" in pipeline.config

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Chroma instance)."""
        pytest.skip("Integration test requires Chroma service")


class TestChromaMultitenancyIndexing:
    """Extended test suite for Chroma indexing pipeline."""

    def test_get_collection_name_with_tenant_suffix(self, tmp_path: Path) -> None:
        """Test _get_collection_name returns name with tenant suffix."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
collection:
  name: test_collection
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            collection_name = pipeline._get_collection_name()
            # Collection name includes tenant suffix for isolation
            assert collection_name == "test_collection_test_tenant"

    def test_get_collection_name_default(self, tmp_path: Path) -> None:
        """Test _get_collection_name returns default value when not in config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            collection_name = pipeline._get_collection_name()
            # Default collection name includes tenant suffix for isolation
            assert collection_name == "multitenancy_test_tenant"

    def test_close_method_indexing(self, tmp_path: Path) -> None:
        """Test close method executes without error."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            # Close should not raise any errors
            pipeline.close()

    def test_run_with_empty_documents(self, tmp_path: Path) -> None:
        """Test run method with empty documents list."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
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
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 0

    def test_load_documents_from_dataloader_applies_params(
        self, tmp_path: Path
    ) -> None:
        """Test dataloader params are applied when loading documents."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
collection:
  name: test_collection
dataloader:
  dataset: custom
  params:
    batch_size: 10
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.DataloaderCatalog"
            ) as mock_catalog,
            patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"),
        ):
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder

            # Setup proper mock chain: loader.load().to_haystack() -> documents
            sample_docs = [Document(content="doc")]
            mock_dataset = MagicMock()
            mock_dataset.to_haystack.return_value = sample_docs
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_catalog.create.return_value = mock_loader

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            docs = pipeline._load_documents_from_dataloader()

            assert docs[0].content == "doc"
            # Verify DataloaderCatalog.create was called with correct params from config
            mock_catalog.create.assert_called_once_with(
                "custom",
                split="test",
                limit=None,
                dataset_id=None,
            )
            mock_loader.load.assert_called_once()
            mock_dataset.to_haystack.assert_called_once()

    def test_run_loads_documents_when_none(self, tmp_path: Path) -> None:
        """Test run loads documents when none are provided."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db_class,
        ):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_db = MagicMock()
            mock_db._collection = mock_collection
            mock_chroma_db_class.return_value = mock_db

            docs = [Document(content="doc", meta={"source": "test"})]
            embedded_docs = [
                Document(content="doc", meta={"source": "test"}, embedding=[0.1] * 1024)
            ]
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": embedded_docs}
            mock_create_embedder.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))

            with patch.object(
                pipeline, "_load_documents_from_dataloader", return_value=docs
            ) as mock_loader:
                result = pipeline.run()

            assert result.documents_indexed == 1
            assert result.tenant_id == "test-tenant"
            mock_loader.assert_called_once()
            mock_db.upsert.assert_called_once()

    def test_run_skips_upsert_when_db_missing(self, tmp_path: Path) -> None:
        """Test run handles missing database client without error."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"),
        ):
            docs = [Document(content="doc", meta={"source": "test"})]
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": docs}
            mock_create_embedder.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            pipeline._db = None
            result = pipeline.run(documents=docs)

            assert result.documents_indexed == 1

    def test_run_with_custom_tenant_id(self, tmp_path: Path) -> None:
        """Test run method with custom tenant_id parameter."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
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
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"documents": []}
            mock_create_embedder.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=[], tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    def test_run_with_documents(self, tmp_path: Path) -> None:
        """Test run method with actual documents."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
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
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db_class,
        ):
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

            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_db = MagicMock()
            mock_db._collection = mock_collection
            mock_chroma_db_class.return_value = mock_db

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            result = pipeline.run(documents=mock_docs)

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 2
            mock_db.upsert.assert_called_once()

    def test_create_timing_metrics_indexing(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db_class,
        ):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_db = MagicMock()
            mock_db._collection = mock_collection
            mock_chroma_db_class.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            metrics = pipeline._create_timing_metrics(
                index_operation_ms=100.0,
                total_ms=100.0,
                num_documents=5,
            )

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test-tenant"
            assert metrics.index_operation_ms == 100.0

    def test_tenant_field_constant(self, tmp_path: Path) -> None:
        """Test that TENANT_FIELD constant is correctly defined."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
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
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(str(config_file))
            assert pipeline.TENANT_FIELD == "tenant_id"

    def test_connect_with_custom_persist_dir(self, tmp_path: Path) -> None:
        """Test _connect method uses custom persist_dir from config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: /custom/path
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            ChromaMultitenancyIndexingPipeline(str(config_file))

            # Verify ChromaVectorDB was called with custom persist_dir
            mock_chroma_db.assert_called_once()
            call_kwargs = mock_chroma_db.call_args.kwargs
            assert call_kwargs.get("persist_dir") == "/custom/path"


class TestChromaMultitenancySearch:
    """Extended test suite for Chroma search pipeline."""

    def test_query_method(self, tmp_path: Path) -> None:
        """Test query method returns results."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
            ) as mock_chroma_db_class,
        ):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_db = MagicMock()
            mock_db._collection = mock_collection
            mock_chroma_db_class.return_value = mock_db

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
            mock_create_embedder.return_value = mock_embedder

            # Mock search result
            mock_db.query.return_value = {
                "ids": [["doc1"]],
                "documents": [["Test content"]],
                "metadatas": [[{"tenant_id": "test-tenant"}]],
                "distances": [[0.1]],
                "embeddings": [[[0.1] * 1024]],
            }
            mock_db.query_to_documents.return_value = [
                Document(
                    content="Test content",
                    meta={"tenant_id": "test-tenant"},
                    embedding=[0.1] * 1024,
                )
            ]

            pipeline = ChromaMultitenancySearchPipeline(str(config_file))
            result = pipeline.query("test query")

            assert result.tenant_id == "test-tenant"
            assert result.query == "test query"

    def test_query_truncates_embeddings(self, tmp_path: Path) -> None:
        """Test query truncates embeddings when output_dimension is set."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
collection:
  name: test_collection
embedding:
  output_dimension: 2
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
            ) as mock_chroma_db_class,
        ):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_db = MagicMock()
            mock_db._collection = mock_collection
            mock_db.query.return_value = {}
            mock_db.query_to_documents.return_value = []
            mock_chroma_db_class.return_value = mock_db

            mock_embedder = MagicMock()
            mock_embedder.run.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_create_embedder.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(str(config_file))
            pipeline.query("test query")

            query_kwargs = mock_db.query.call_args.kwargs
            assert query_kwargs["query_embedding"] == [0.1, 0.2]

    def test_connect_enables_rag_pipeline(self, tmp_path: Path) -> None:
        """Test RAG pipeline is initialized when enabled in config."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
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
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_create_embedder,
            patch("vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"),
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_rag_pipeline"
            ) as mock_create_rag_pipeline,
        ):
            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder
            mock_rag_pipeline = MagicMock()
            mock_create_rag_pipeline.return_value = mock_rag_pipeline

            pipeline = ChromaMultitenancySearchPipeline(str(config_file))

            assert pipeline._rag_pipeline is mock_rag_pipeline
            mock_create_rag_pipeline.assert_called_once()

    def test_rag_runs_pipeline(self, tmp_path: Path) -> None:
        """Test RAG flow uses retrieval results and returns response."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
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
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_create_embedder,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
            ) as mock_chroma_db,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_rag_pipeline"
            ) as mock_create_rag,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db

            mock_embedder = MagicMock()
            mock_create_embedder.return_value = mock_embedder
            mock_rag = MagicMock()
            mock_rag.run.return_value = {"generator": {"replies": ["answer"]}}
            mock_create_rag.return_value = mock_rag

            pipeline = ChromaMultitenancySearchPipeline(str(config_file))

            retrieval_docs = [Document(content="doc", meta={})]
            retrieval_result = SimpleNamespace(documents=retrieval_docs, scores=[0.5])
            with patch.object(
                pipeline, "query", return_value=retrieval_result
            ) as mock_query:
                result = pipeline.rag("test query", top_k=2)

            assert result.generated_response == "answer"
            assert result.retrieved_documents == retrieval_docs
            assert result.retrieval_scores == [0.5]
            mock_query.assert_called_once_with("test query", 2, "test-tenant")

    def test_close_method_search(self, tmp_path: Path) -> None:
        """Test close method executes without error."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
collection:
  name: test_collection
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
            ) as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(str(config_file))
            # Close should not raise any errors
            pipeline.close()

    def test_create_timing_metrics_search(self, tmp_path: Path) -> None:
        """Test _create_timing_metrics method."""
        config_file = tmp_path / "config.yaml"
        config_content = """
chroma:
  persist_dir: ./test_db
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
            ) as mock_chroma_db_class,
        ):
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_db = MagicMock()
            mock_db._collection = mock_collection
            mock_chroma_db_class.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(str(config_file))
            # Search pipeline's _create_timing_metrics takes just total_ms
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test-tenant"
