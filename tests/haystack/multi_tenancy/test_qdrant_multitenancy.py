"""Tests for Qdrant multi-tenancy pipelines."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


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

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    def test_indexing_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test that Qdrant indexing pipeline initializes correctly."""
        qdrant_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = qdrant_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

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

            pipeline = QdrantMultitenancyIndexingPipeline(qdrant_config)
            assert pipeline.config == qdrant_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    def test_search_pipeline_initialization(
        self,
        mock_load_config: MagicMock,
        mock_resolve_context: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test that Qdrant search pipeline initializes correctly."""
        qdrant_config["tenant"] = {"id": "test_tenant"}
        mock_load_config.return_value = qdrant_config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

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

            pipeline = QdrantMultitenancySearchPipeline(qdrant_config)
            assert pipeline.config == qdrant_config
            assert pipeline.tenant_context.tenant_id == "test_tenant"

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_end_to_end_flow(self) -> None:
        """Integration test for end-to-end flow (requires Qdrant instance)."""
        pytest.skip("Integration test requires Qdrant service")

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    def test_get_collection_name_from_config(
        self, mock_load_config, mock_resolve_context
    ):
        """Test _get_collection_name returns value from config."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "collection": {"name": "custom-collection"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

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

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            collection_name = pipeline._get_collection_name()
            assert collection_name == "custom-collection"

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    def test_get_collection_name_default(self, mock_load_config, mock_resolve_context):
        """Test _get_collection_name returns default value when not in config."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

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

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            collection_name = pipeline._get_collection_name()
            assert collection_name == "multitenancy"

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    def test_close_method(self, mock_load_config, mock_resolve_context):
        """Test close method calls client.close()."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

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

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            pipeline.close()
            mock_client.close.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder")
    def test_run_with_empty_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with empty documents list."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with (
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[])

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 0

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder")
    def test_run_with_custom_tenant_id(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with custom tenant_id parameter."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "original_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="original_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": []}
        mock_create_embedder.return_value = mock_embedder

        with (
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=[], tenant_id="custom_tenant")

            assert result.tenant_id == "custom_tenant"

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder")
    def test_run_with_documents(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run method with actual documents."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
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

        with (
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            result = pipeline.run(documents=mock_docs)

            assert isinstance(result, TenantIndexResult)
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 2
            mock_client.upsert.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    def test_create_timing_metrics(self, mock_load_config, mock_resolve_context):
        """Test _create_timing_metrics method."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

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

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            metrics = pipeline._create_timing_metrics(5, 100.0)

            assert metrics.num_documents == 5
            assert metrics.total_ms == 100.0
            assert metrics.tenant_id == "test_tenant"
            assert metrics.index_operation_ms == 100.0

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder")
    def test_connect_with_local_location(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _connect method with local location config."""
        config = {
            "qdrant": {"location": "./test_data"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = False
            mock_qdrant_client.return_value = mock_client

            QdrantMultitenancyIndexingPipeline(config)

            # Verify QdrantClient was called with location
            mock_qdrant_client.assert_called_with(location="./test_data")
            mock_client.create_collection.assert_called_once()
            mock_client.create_payload_index.assert_called_once()

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder")
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

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with (
            patch.dict(
                "os.environ", {"QDRANT_HOST": "remote-host", "QDRANT_PORT": "6334"}
            ),
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

            QdrantMultitenancyIndexingPipeline(config)

            # Verify QdrantClient was called with host and port from env
            mock_qdrant_client.assert_called_with(host="remote-host", port=6334)

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.DataloaderCatalog")
    def test_load_documents_from_dataloader_with_params(
        self,
        mock_catalog_class,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _load_documents_from_dataloader applies params from config."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
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

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Setup new API mock pattern
        mock_docs = [
            Document(content="Doc from dataloader", meta={"source": "triviaqa"})
        ]
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = mock_docs

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        mock_catalog_class.create.return_value = mock_loader

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            documents = pipeline._load_documents_from_dataloader()

            # Verify DataloaderCatalog.create was called correctly
            mock_catalog_class.create.assert_called_once_with(
                "triviaqa",
                split="train",
                limit=None,
                dataset_id=None,
            )
            # Verify documents returned
            assert len(documents) == 1
            assert documents[0].content == "Doc from dataloader"

    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.qdrant.indexing.DataloaderCatalog")
    def test_run_with_documents_none_triggers_dataloader(
        self,
        mock_catalog_class,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test run() with documents=None loads from dataloader."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"dimension": 1024},
            "dataloader": {
                "dataset": "popqa",
            },
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.indexing import (
            QdrantMultitenancyIndexingPipeline,
        )

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

        # Setup new API mock pattern
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = mock_docs_from_loader

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        mock_catalog_class.create.return_value = mock_loader

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_qdrant_client.return_value = mock_client

            pipeline = QdrantMultitenancyIndexingPipeline(config)
            # Call run with documents=None to trigger dataloader
            result = pipeline.run(documents=None)

            # Verify dataloader was used
            mock_catalog_class.create.assert_called_once_with(
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
            assert result.tenant_id == "test_tenant"
            assert result.documents_indexed == 2

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_query_method(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method returns results."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            # Mock search result - Qdrant returns a list of ScoredPoint
            mock_result = MagicMock()
            mock_result.payload = {
                "content": "Test content",
                "tenant_id": "test_tenant",
            }
            mock_result.score = 0.95
            mock_client.search.return_value = [mock_result]

            pipeline = QdrantMultitenancySearchPipeline(config)
            result = pipeline.query("test query")

            assert result.tenant_id == "test_tenant"
            assert result.query == "test query"
            assert len(result.documents) == 1
            assert result.documents[0].content == "Test content"

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_query_with_custom_top_k(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method with custom top_k parameter."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            mock_result = MagicMock()
            mock_result.points = []
            mock_client.search.return_value = mock_result

            pipeline = QdrantMultitenancySearchPipeline(config)
            pipeline.query("test query", top_k=20)

            # Verify search was called with correct top_k
            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs.get(
                "limit", call_kwargs.get("top", call_kwargs.get("top_k", 0))
            ) in [20, None]

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_query_with_custom_tenant_id(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method with custom tenant_id parameter."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "original_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="original_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            mock_result = MagicMock()
            mock_result.points = []
            mock_client.search.return_value = mock_result

            pipeline = QdrantMultitenancySearchPipeline(config)
            result = pipeline.query("test query", tenant_id="custom_tenant")

            assert result.tenant_id == "custom_tenant"

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    def test_search_close_method(self, mock_load_config, mock_resolve_context):
        """Test close method calls client.close()."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

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

            pipeline = QdrantMultitenancySearchPipeline(config)
            pipeline.close()
            mock_client.close.assert_called_once()

    def test_search_create_timing_metrics(self):
        """Test _create_timing_metrics method."""
        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_config = {
            "qdrant": {"url": "http://localhost:6333"},
        }

        with (
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.load_config",
                return_value=mock_config,
            ),
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve",
                return_value=TenantContext("test_tenant"),
            ),
            patch(
                "vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder"
            ) as mock_embedder_creator,
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_embedder = MagicMock()
            mock_embedder_creator.return_value = mock_embedder

            pipeline = QdrantMultitenancySearchPipeline(mock_config)
            metrics = pipeline._create_timing_metrics(50.0)

            assert metrics.retrieval_ms == 50.0
            assert metrics.total_ms == 50.0
            assert metrics.tenant_id == "test_tenant"

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_query_empty_results(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method returns empty results when no matches."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            mock_result = MagicMock()
            mock_result.points = []
            mock_client.search.return_value = mock_result

            pipeline = QdrantMultitenancySearchPipeline(config)
            result = pipeline.query("test query")

            assert result.tenant_id == "test_tenant"
            assert len(result.documents) == 0
            assert len(result.scores) == 0

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_search_connect_with_local_location(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _connect method with local location config."""
        config = {
            "qdrant": {"location": "./test_data"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            QdrantMultitenancySearchPipeline(config)

            # Verify QdrantClient was called with location
            mock_qdrant_client.assert_called_with(location="./test_data")

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_search_connect_with_env_vars(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test _connect method uses environment variables when config not present."""
        config = {
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with (
            patch.dict(
                "os.environ", {"QDRANT_HOST": "remote-host", "QDRANT_PORT": "6334"}
            ),
            patch("qdrant_client.QdrantClient") as mock_qdrant_client,
        ):
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            QdrantMultitenancySearchPipeline(config)

            # Verify QdrantClient was called with host and port from env
            mock_qdrant_client.assert_called_with(host="remote-host", port=6334)

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    def test_close_with_none_client(self, mock_load_config, mock_resolve_context):
        """Test close method when _client is None (no-op)."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

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

            pipeline = QdrantMultitenancySearchPipeline(config)
            # Manually set _client to None to simulate uninitialized state
            pipeline._client = None
            # Should not raise and should not call close on None
            pipeline.close()

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_query_with_output_dimension_truncation(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test query method truncates embedding when output_dimension is specified."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "embedding": {"output_dimension": 512},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        # Create a 1024-dimensional embedding that should be truncated to 512
        full_embedding = list(range(1024))
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": full_embedding}
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client
            mock_client.search.return_value = []

            pipeline = QdrantMultitenancySearchPipeline(config)
            pipeline.query("test query")

            # Verify search was called with truncated embedding (512 dimensions)
            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            query_vector = call_kwargs["query_vector"]
            assert len(query_vector) == 512
            assert query_vector == list(range(512))

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    def test_rag_raises_when_not_enabled(
        self,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test rag method raises ValueError when RAG pipeline is not enabled."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            pipeline = QdrantMultitenancySearchPipeline(config)

            with pytest.raises(ValueError, match="RAG pipeline not enabled/configured"):
                pipeline.rag("test query")

    @patch("vectordb.haystack.multi_tenancy.qdrant.search.TenantContext.resolve")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.load_config")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_text_embedder")
    @patch("vectordb.haystack.multi_tenancy.qdrant.search.create_rag_pipeline")
    def test_rag_success_when_enabled(
        self,
        mock_create_rag_pipeline,
        mock_create_embedder,
        mock_load_config,
        mock_resolve_context,
    ):
        """Test rag method succeeds when RAG pipeline is enabled."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "tenant": {"id": "test_tenant"},
            "collection": {"name": "test_collection"},
            "rag": {"enabled": True},
        }
        mock_load_config.return_value = config
        mock_tenant_ctx = TenantContext(tenant_id="test_tenant")
        mock_resolve_context.return_value = mock_tenant_ctx

        from vectordb.haystack.multi_tenancy.qdrant.search import (
            QdrantMultitenancySearchPipeline,
        )

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 1024}
        mock_create_embedder.return_value = mock_embedder

        # Mock the RAG pipeline
        mock_rag_pipeline = MagicMock()
        mock_rag_pipeline.run.return_value = {
            "generator": {"replies": ["Generated response from RAG"]}
        }
        mock_create_rag_pipeline.return_value = mock_rag_pipeline

        with patch("qdrant_client.QdrantClient") as mock_qdrant_client:
            mock_client = MagicMock()
            mock_qdrant_client.return_value = mock_client

            # Mock search result for query
            mock_result = MagicMock()
            mock_result.payload = {
                "content": "Test document content",
                "tenant_id": "test_tenant",
            }
            mock_result.score = 0.95
            mock_client.search.return_value = [mock_result]

            pipeline = QdrantMultitenancySearchPipeline(config)
            result = pipeline.rag("test query", top_k=5, tenant_id="test_tenant")

            # Verify the result
            assert result.tenant_id == "test_tenant"
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
