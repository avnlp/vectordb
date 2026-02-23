"""Tests for Weaviate namespace indexing pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestWeaviateNamespaceIndexingPipeline:
    """Unit tests for Weaviate namespace indexing pipeline."""

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_valid_namespace(
        self,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test pipeline initialization with valid namespace."""
        mock_embedder.return_value = MagicMock()

        from vectordb.langchain.namespaces.indexing.weaviate import (
            WeaviateNamespaceIndexingPipeline,
        )

        pipeline = WeaviateNamespaceIndexingPipeline(
            weaviate_namespace_config, "ns_123"
        )

        assert pipeline.namespace == "ns_123"
        assert pipeline.config == weaviate_namespace_config

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_empty_namespace_raises_error(
        self,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test initialization with empty namespace raises ValueError."""
        from vectordb.langchain.namespaces.indexing.weaviate import (
            WeaviateNamespaceIndexingPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            WeaviateNamespaceIndexingPipeline(weaviate_namespace_config, "")

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_run_returns_indexed_count(
        self,
        mock_embed_documents: MagicMock,
        mock_get_documents: MagicMock,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test run returns correct indexed document count."""
        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db.list_tenants.return_value = ["ns_123"]
        mock_db_cls.return_value = mock_db

        mock_embedder.return_value = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
            Document(page_content="doc3", metadata={"id": "3"}),
        ]
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (
            ["doc1", "doc2", "doc3"],
            [[0.1] * 384] * 3,
        )

        from vectordb.langchain.namespaces.indexing.weaviate import (
            WeaviateNamespaceIndexingPipeline,
        )

        pipeline = WeaviateNamespaceIndexingPipeline(
            weaviate_namespace_config, "ns_123"
        )
        result = pipeline.run()

        assert "documents_indexed" in result
        assert result["namespace"] == "ns_123"
        assert result["documents_indexed"] == 3

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    def test_run_with_no_documents_returns_zero(
        self,
        mock_get_documents: MagicMock,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test run with no documents returns 0 indexed."""
        mock_embedder.return_value = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        from vectordb.langchain.namespaces.indexing.weaviate import (
            WeaviateNamespaceIndexingPipeline,
        )

        pipeline = WeaviateNamespaceIndexingPipeline(
            weaviate_namespace_config, "ns_123"
        )
        result = pipeline.run()

        assert result["documents_indexed"] == 0
