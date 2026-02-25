"""Tests for Qdrant multi-tenancy indexing pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestQdrantMultiTenancyIndexingPipeline:
    """Unit tests for Qdrant multi-tenancy indexing pipeline."""

    @patch("vectordb.langchain.multi_tenancy.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_valid_tenant_id(
        self, mock_embedder, mock_db_cls, qdrant_multi_tenant_config: dict
    ):
        """Test pipeline initialization with valid tenant ID."""
        mock_embedder.return_value = MagicMock()

        from vectordb.langchain.multi_tenancy.indexing.qdrant import (
            QdrantMultiTenancyIndexingPipeline,
        )

        pipeline = QdrantMultiTenancyIndexingPipeline(
            qdrant_multi_tenant_config, "tenant_123"
        )

        assert pipeline.tenant_id == "tenant_123"
        assert pipeline.config == qdrant_multi_tenant_config

    @patch("vectordb.langchain.multi_tenancy.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_empty_tenant_id_raises_error(
        self, mock_embedder, mock_db_cls, qdrant_multi_tenant_config: dict
    ):
        """Test initialization with empty tenant_id raises ValueError."""
        from vectordb.langchain.multi_tenancy.indexing.qdrant import (
            QdrantMultiTenancyIndexingPipeline,
        )

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            QdrantMultiTenancyIndexingPipeline(qdrant_multi_tenant_config, "")

    @patch("vectordb.langchain.multi_tenancy.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_run_returns_indexed_count(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        qdrant_multi_tenant_config: dict,
    ):
        """Test run returns correct indexed document count."""
        mock_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = ([], [])

        from vectordb.langchain.multi_tenancy.indexing.qdrant import (
            QdrantMultiTenancyIndexingPipeline,
        )

        pipeline = QdrantMultiTenancyIndexingPipeline(
            qdrant_multi_tenant_config, "tenant_123"
        )
        result = pipeline.run()

        assert "documents_indexed" in result
        assert result["tenant_id"] == "tenant_123"
        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.multi_tenancy.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    def test_run_with_no_documents_returns_zero(
        self,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        qdrant_multi_tenant_config: dict,
    ):
        """Test run with no documents returns 0 indexed."""
        mock_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        from vectordb.langchain.multi_tenancy.indexing.qdrant import (
            QdrantMultiTenancyIndexingPipeline,
        )

        pipeline = QdrantMultiTenancyIndexingPipeline(
            qdrant_multi_tenant_config, "tenant_123"
        )
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.multi_tenancy.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_uses_tenant_collection(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        qdrant_multi_tenant_config: dict,
    ):
        """Test indexing uses tenant-specific collection."""
        mock_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = [
            Document(page_content="doc1", metadata={"id": "1"})
        ]
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (["doc1"], [[0.1] * 384])

        from vectordb.langchain.multi_tenancy.indexing.qdrant import (
            QdrantMultiTenancyIndexingPipeline,
        )

        pipeline = QdrantMultiTenancyIndexingPipeline(
            qdrant_multi_tenant_config, "tenant_abc"
        )
        result = pipeline.run()

        # Verify tenant_id is correct in result
        assert result["tenant_id"] == "tenant_abc"

    @patch("vectordb.langchain.multi_tenancy.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_with_limit_respects_limit(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        qdrant_multi_tenant_config: dict,
    ):
        """Test indexing respects dataloader limit."""
        mock_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = ([], [])

        # Add limit to config
        qdrant_multi_tenant_config["dataloader"]["limit"] = 2

        from vectordb.langchain.multi_tenancy.indexing.qdrant import (
            QdrantMultiTenancyIndexingPipeline,
        )

        pipeline = QdrantMultiTenancyIndexingPipeline(
            qdrant_multi_tenant_config, "tenant_123"
        )
        pipeline.run()

        # Verify DataloaderCatalog.create was called with the correct parameters
        mock_get_documents.assert_called_once_with("arc", split="test", limit=2)
