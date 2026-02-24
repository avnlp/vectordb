"""Tests for Milvus multi-tenancy pipelines."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


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

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test that Milvus indexing pipeline initializes correctly."""
        milvus_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = milvus_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

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

            pipeline = MilvusMultitenancyIndexingPipeline(milvus_config)
            assert pipeline.config == milvus_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test that Milvus search pipeline initializes correctly."""
        milvus_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = milvus_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

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

            pipeline = MilvusMultitenancySearchPipeline(milvus_config)
            assert pipeline.config == milvus_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Milvus instance)."""
        pytest.skip("Integration test requires Milvus service")


class TestMilvusMultitenancyIndexing:
    """Extended test suite for Milvus indexing pipeline."""

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    def test_get_collection_name_from_config(
        self, mock_load_config, mock_resolve_context
    ):
        """Test _get_collection_name returns value from config."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "collection": {"name": "custom-collection"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

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

            pipeline = MilvusMultitenancyIndexingPipeline(config)
            collection_name = pipeline._get_collection_name()
            assert collection_name == "custom-collection"

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    def test_get_collection_name_default(self, mock_load_config, mock_resolve_context):
        """Test _get_collection_name returns default value when not in config."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

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

            pipeline = MilvusMultitenancyIndexingPipeline(config)
            collection_name = pipeline._get_collection_name()
            assert collection_name == "multitenancy"

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    def test_close_method(self, mock_load_config, mock_resolve_context):
        """Test close method calls db.close()."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

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

            pipeline = MilvusMultitenancyIndexingPipeline(config)
            pipeline.close()
            mock_db.close.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder")
    def test_run_with_empty_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with empty documents list."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 0

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder")
    def test_run_with_custom_tenant_id(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with custom tenant_id parameter."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "original_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="original_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": []}
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[], tenant_id="custom_tenant")

            assert result.tenant_id == "custom_tenant"

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder")
    def test_run_with_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with actual documents."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
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

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=mock_docs)

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 2
            mock_db.insert_documents.assert_called_once()

    def test_create_timing_metrics(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

        mock_config = {
            "milvus": {"uri": "http://localhost:19530"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve",
                return_value=TenantContext("test_tenant"),
            ),
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

            pipeline = MilvusMultitenancyIndexingPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(5, 100.0)

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test_tenant"
            assert metrics.index_operation_ms == 100.0

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder")
    def test_connect_with_env_vars(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _connect method uses environment variables when config not present."""
        config = {
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with (
            patch.dict(
                "os.environ",
                {"MILVUS_URI": "http://env-host:19530", "MILVUS_TOKEN": "test-token"},
            ),
            patch(
                "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
            ) as mock_milvus_db,
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            MilvusMultitenancyIndexingPipeline(config)

            # Verify MilvusVectorDB was called with URI from env
            mock_milvus_db.assert_called_once()
            call_kwargs = mock_milvus_db.call_args.kwargs
            assert call_kwargs.get("uri") == "http://env-host:19530"
            assert call_kwargs.get("token") == "test-token"

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder")
    def test_tenant_field_constant(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test that TENANT_FIELD constant is correctly defined."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancyIndexingPipeline(config)
            assert pipeline.TENANT_FIELD == "tenant_id"

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.DataloaderCatalog")
    def test_load_documents_from_dataloader_with_params(
        self,
        mock_catalog,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _load_documents_from_dataloader sets attributes from params config."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
            "dataloader": {
                "dataset": "triviaqa",
                "params": {
                    "split": "train",
                    "max_samples": 100,
                },
            },
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

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

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancyIndexingPipeline(config)
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

    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.milvus.indexing.DataloaderCatalog")
    def test_run_with_documents_none_uses_dataloader(
        self,
        mock_catalog,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run() with documents=None loads from dataloader."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
            "dataloader": {
                "dataset": "popqa",
                "params": {"limit": 50},
            },
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.indexing import (
            MilvusMultitenancyIndexingPipeline,
        )

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

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.indexing.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancyIndexingPipeline(config)
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

    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_rag_pipeline")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_connect_warms_embedder_and_sets_rag_pipeline(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
        mock_create_rag_pipeline,
    ):
        """Test connect initializes embedder and RAG pipeline when enabled."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "rag": {"enabled": True},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test_tenant")

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        mock_rag_pipeline = MagicMock()
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancySearchPipeline(config)

        mock_embedder.warm_up.assert_called_once()
        mock_create_rag_pipeline.assert_called_once_with(config)
        assert pipeline._rag_pipeline == mock_rag_pipeline

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_query_method(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method returns results."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            # Mock search result
            mock_db.retrieve.return_value = [
                Document(
                    content="Test content",
                    meta={"tenant_id": "test_tenant"},
                    embedding=[0.1] * 1024,
                )
            ]

            pipeline = MilvusMultitenancySearchPipeline(config)
            result = pipeline.query("test query")

            assert result.tenant_id == "test_tenant"
            assert result.query == "test query"

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_query_truncates_embedding_when_output_dimension_set(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query truncates embeddings when output dimension is configured."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"output_dimension": 4},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test_tenant")

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        mock_create_embedder.return_value = mock_embedder

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_db.retrieve.return_value = []
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancySearchPipeline(config)
            pipeline.query("test query")

        call_kwargs = mock_db.retrieve.call_args.kwargs
        assert call_kwargs["query_embedding"] == [0.1, 0.2, 0.3, 0.4]

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_query_with_custom_top_k(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method with custom top_k parameter."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_db.retrieve.return_value = []

            pipeline = MilvusMultitenancySearchPipeline(config)
            pipeline.query("test query", top_k=20)

            # Verify retrieve was called
            mock_db.retrieve.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_query_with_custom_tenant_id(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method with custom tenant_id parameter."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "original_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="original_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db
            mock_db.retrieve.return_value = []

            pipeline = MilvusMultitenancySearchPipeline(config)
            result = pipeline.query("test query", tenant_id="custom_tenant")

            assert result.tenant_id == "custom_tenant"

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    def test_close_method(self, mock_load_config, mock_resolve_context):
        """Test close method calls db.close()."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

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

            pipeline = MilvusMultitenancySearchPipeline(config)
            pipeline.close()
            mock_db.close.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_close_no_db_connection(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test close handles missing db connection gracefully."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test_tenant")

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancySearchPipeline(config)
            pipeline._db = None
            pipeline.close()

    def test_create_timing_metrics(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        mock_config = {
            "milvus": {"uri": "http://localhost:19530"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve",
                return_value=TenantContext("test_tenant"),
            ),
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

            pipeline = MilvusMultitenancySearchPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test_tenant"

    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_rag_disabled_raises_error(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test rag raises when pipeline is not configured/enabled."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "rag": {"enabled": False},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test_tenant")

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        with (
            patch(
                "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
            ) as mock_milvus_db,
            pytest.raises(ValueError, match="RAG pipeline not enabled/configured"),
        ):
            mock_db = MagicMock()
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancySearchPipeline(config)
            pipeline.rag("test query")

    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_rag_pipeline")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.milvus.search.create_text_embedder")
    def test_rag_returns_generated_response(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
        mock_create_rag_pipeline,
    ):
        """Test rag returns generated response with retrieved documents."""
        config = {
            "milvus": {"uri": "http://localhost:19530"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "rag": {"enabled": True},
        }
        mock_load_config.return_value = config
        mock_resolve_context.return_value = TenantContext(tenant_id="test_tenant")

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 8}
        mock_create_embedder.return_value = mock_embedder

        mock_rag_pipeline = MagicMock()
        mock_rag_pipeline.run.return_value = {
            "generator": {"replies": ["Generated answer"]}
        }
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        from vectordb.haystack.multi_tenancy.milvus.search import (
            MilvusMultitenancySearchPipeline,
        )

        with patch(
            "vectordb.haystack.multi_tenancy.milvus.search.MilvusVectorDB"
        ) as mock_milvus_db:
            mock_db = MagicMock()
            mock_db.retrieve.return_value = [
                Document(content="Result", meta={"tenant_id": "test_tenant"})
            ]
            mock_milvus_db.return_value = mock_db

            pipeline = MilvusMultitenancySearchPipeline(config)
            result = pipeline.rag("test query", top_k=3)

        assert result.generated_response == "Generated answer"
        mock_rag_pipeline.run.assert_called_once()
