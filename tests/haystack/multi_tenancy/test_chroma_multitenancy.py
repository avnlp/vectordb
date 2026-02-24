"""Tests for Chroma multi-tenancy pipelines."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


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

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test that Chroma indexing pipeline initializes correctly."""
        chroma_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = chroma_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"),
        ):
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(chroma_config)
            assert pipeline.config == chroma_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @patch("vectordb.haystack.multi_tenancy.chroma.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test that Chroma search pipeline initializes correctly."""
        chroma_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = chroma_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.search import (
            ChromaMultitenancySearchPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"),
        ):
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(chroma_config)
            assert pipeline.config == chroma_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Chroma instance)."""
        pytest.skip("Integration test requires Chroma service")


class TestChromaMultitenancyIndexing:
    """Extended test suite for Chroma indexing pipeline."""

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    def test_get_collection_name_with_tenant_suffix(
        self, mock_load_config, mock_resolve_context
    ):
        """Test _get_collection_name returns name with tenant suffix."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "collection": {"name": "test_collection"},
            "tenant": {"id": "test-tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(config)
            collection_name = pipeline._get_collection_name()
            assert collection_name == "test_collection_test_tenant"

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    def test_get_collection_name_default(self, mock_load_config, mock_resolve_context):
        """Test _get_collection_name returns default value when not in config."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(config)
            collection_name = pipeline._get_collection_name()
            assert collection_name == "multitenancy_test_tenant"

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    def test_close_method_indexing(self, mock_load_config, mock_resolve_context):
        """Test close method executes without error."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(config)
            # Close should not raise any errors
            pipeline.close()

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder")
    def test_run_with_empty_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with empty documents list."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db:
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db

            pipeline = ChromaMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test-tenant"
            assert result.documents_indexed == 0

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.DataloaderCatalog")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder")
    def test_load_documents_from_dataloader_applies_params(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
        mock_catalog,
    ):
        """Test dataloader params are applied when loading documents."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
            "dataloader": {
                "dataset": "custom",
                "params": {"batch_size": 10},
            },
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test-tenant")

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Setup proper mock chain: loader.load().to_haystack() -> documents
        sample_docs = [Document(content="doc")]
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_docs
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_catalog.create.return_value = mock_loader

        with patch("vectordb.databases.chroma.ChromaVectorDB"):
            pipeline = ChromaMultitenancyIndexingPipeline(config)
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

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder")
    def test_run_loads_documents_when_none(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run loads documents when none are provided."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test-tenant")

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        docs = [Document(content="doc", meta={"source": "test"})]
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": docs}
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"
        ) as mock_chroma_db:
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            pipeline = ChromaMultitenancyIndexingPipeline(config)

            with patch.object(
                pipeline, "_load_documents_from_dataloader", return_value=docs
            ) as mock_loader:
                result = pipeline.run()

        assert result.documents_indexed == 1
        assert result.tenant_id == "test-tenant"
        mock_loader.assert_called_once()
        mock_db.upsert.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder")
    def test_run_skips_upsert_when_db_missing(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run handles missing database client without error."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test-tenant")

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        docs = [Document(content="doc", meta={"source": "test"})]
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": docs}
        mock_create_embedder.return_value = mock_embedder

        with patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB"):
            pipeline = ChromaMultitenancyIndexingPipeline(config)
            pipeline._db = None
            result = pipeline.run(documents=docs)

        assert result.documents_indexed == 1

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder")
    def test_run_with_custom_tenant_id(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with custom tenant_id parameter."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "original-tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="original-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": []}
        mock_create_embedder.return_value = mock_embedder

        with patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db:
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db

            pipeline = ChromaMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[], tenant_id="custom-tenant")

            assert result.tenant_id == "custom-tenant"

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.ChromaVectorDB")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder")
    def test_run_with_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
        mock_chroma_db,
    ):
        """Test run method with actual documents."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
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

        mock_db = MagicMock()
        mock_chroma_db.return_value = mock_db

        pipeline = ChromaMultitenancyIndexingPipeline(config)
        result = pipeline.run(documents=mock_docs)

        assert isinstance(result, TenantIndexResult)
        assert result.tenant_id == "test-tenant"
        assert result.documents_indexed == 2
        mock_db.upsert.assert_called_once()

    def test_create_timing_metrics_indexing(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        mock_config = {
            "chroma": {"persist_dir": "./test_db"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve",
                return_value=TenantContext("test-tenant"),
            ),
            patch(
                "vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancyIndexingPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(5, 100.0)

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test-tenant"
            assert metrics.index_operation_ms == 100.0

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.create_document_embedder")
    def test_tenant_field_constant(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test that TENANT_FIELD constant is correctly defined."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db:
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db

            pipeline = ChromaMultitenancyIndexingPipeline(config)
            assert pipeline.TENANT_FIELD == "tenant_id"

    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.indexing.load_config")
    def test_connect_with_custom_persist_dir(
        self, mock_load_config, mock_resolve_context
    ):
        """Test _connect method uses custom persist_dir from config."""
        config = {
            "chroma": {"persist_dir": "/custom/path"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.indexing import (
            ChromaMultitenancyIndexingPipeline,
        )

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

            ChromaMultitenancyIndexingPipeline(config)

            # Verify ChromaVectorDB was called with custom persist_dir
            mock_chroma_db.assert_called_once()
            call_kwargs = mock_chroma_db.call_args.kwargs
            assert call_kwargs.get("persist_dir") == "/custom/path"


class TestChromaMultitenancySearch:
    """Extended test suite for Chroma search pipeline."""

    @patch("vectordb.haystack.multi_tenancy.chroma.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder")
    def test_query_method(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method returns results."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.search import (
            ChromaMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
        ) as mock_chroma_db:
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db

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

            pipeline = ChromaMultitenancySearchPipeline(config)
            result = pipeline.query("test query")

            assert result.tenant_id == "test-tenant"
            assert result.query == "test query"

    @patch("vectordb.haystack.multi_tenancy.chroma.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder")
    def test_query_truncates_embeddings(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query truncates embeddings when output_dimension is set."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"output_dimension": 2},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test-tenant")

        from vectordb.haystack.multi_tenancy.chroma.search import (
            ChromaMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
        ) as mock_chroma_db:
            mock_db = MagicMock()
            mock_db.query.return_value = {}
            mock_db.query_to_documents.return_value = []
            mock_chroma_db.return_value = mock_db

            pipeline = ChromaMultitenancySearchPipeline(config)
            pipeline.query("test query")

        query_kwargs = mock_db.query.call_args.kwargs
        assert query_kwargs["query_embedding"] == [0.1, 0.2]

    @patch("vectordb.haystack.multi_tenancy.chroma.search.create_rag_pipeline")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder")
    def test_connect_enables_rag_pipeline(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
        mock_create_rag_pipeline,
    ):
        """Test RAG pipeline is initialized when enabled in config."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
            "rag": {"enabled": True},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test-tenant")

        from vectordb.haystack.multi_tenancy.chroma.search import (
            ChromaMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder
        mock_rag_pipeline = MagicMock()
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        with patch("vectordb.databases.chroma.ChromaVectorDB"):
            pipeline = ChromaMultitenancySearchPipeline(config)

        assert pipeline._rag_pipeline is mock_rag_pipeline
        mock_create_rag_pipeline.assert_called_once_with(config)

    @patch("vectordb.haystack.multi_tenancy.chroma.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder")
    def test_rag_runs_pipeline(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test RAG flow uses retrieval results and returns response."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
            "rag": {"enabled": True},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test-tenant")

        from vectordb.haystack.multi_tenancy.chroma.search import (
            ChromaMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder
        mock_rag = MagicMock()
        mock_rag.run.return_value = {"generator": {"replies": ["answer"]}}

        with patch(
            "vectordb.haystack.multi_tenancy.chroma.search.ChromaVectorDB"
        ) as mock_chroma_db:
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db

            with patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_rag_pipeline",
                return_value=mock_rag,
            ):
                pipeline = ChromaMultitenancySearchPipeline(config)

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

    @patch("vectordb.haystack.multi_tenancy.chroma.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.chroma.search.load_config")
    def test_close_method_search(self, mock_load_config, mock_resolve_context):
        """Test close method executes without error."""
        config = {
            "chroma": {"persist_dir": "./test_db"},
            "tenant": {"id": "test-tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test-tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.chroma.search import (
            ChromaMultitenancySearchPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(config)
            # Close should not raise any errors
            pipeline.close()

    def test_create_timing_metrics_search(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.chroma.search import (
            ChromaMultitenancySearchPipeline,
        )

        mock_config = {
            "chroma": {"persist_dir": "./test_db"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.TenantContext.resolve",
                return_value=TenantContext("test-tenant"),
            ),
            patch(
                "vectordb.haystack.multi_tenancy.chroma.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("vectordb.databases.chroma.ChromaVectorDB") as mock_chroma_db,
        ):
            mock_db = MagicMock()
            mock_chroma_db.return_value = mock_db
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = ChromaMultitenancySearchPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test-tenant"
