"""Tests for Pinecone namespace indexing pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestPineconeNamespaceIndexingPipeline:
    """Unit tests for Pinecone namespace indexing pipeline."""

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_valid_namespace(
        self, mock_embedder, mock_db_cls, pinecone_namespace_config: dict
    ):
        """Test pipeline initialization with valid namespace."""
        mock_embedder.return_value = MagicMock()

        from vectordb.langchain.namespaces.indexing.pinecone import (
            PineconeNamespaceIndexingPipeline,
        )

        pipeline = PineconeNamespaceIndexingPipeline(
            pinecone_namespace_config, "ns_123"
        )

        assert pipeline.namespace == "ns_123"
        assert pipeline.config == pinecone_namespace_config

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_empty_namespace_raises_error(
        self, mock_embedder, mock_db_cls, pinecone_namespace_config: dict
    ):
        """Test initialization with empty namespace raises ValueError."""
        from vectordb.langchain.namespaces.indexing.pinecone import (
            PineconeNamespaceIndexingPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            PineconeNamespaceIndexingPipeline(pinecone_namespace_config, "")

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_run_returns_indexed_count(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
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

        from vectordb.langchain.namespaces.indexing.pinecone import (
            PineconeNamespaceIndexingPipeline,
        )

        pipeline = PineconeNamespaceIndexingPipeline(
            pinecone_namespace_config, "ns_123"
        )
        result = pipeline.run()

        assert "documents_indexed" in result
        assert result["namespace"] == "ns_123"
        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    def test_run_with_no_documents_returns_zero(
        self,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
    ):
        """Test run with no documents returns 0 indexed."""
        mock_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        from vectordb.langchain.namespaces.indexing.pinecone import (
            PineconeNamespaceIndexingPipeline,
        )

        pipeline = PineconeNamespaceIndexingPipeline(
            pinecone_namespace_config, "ns_123"
        )
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_uses_namespace(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
    ):
        """Test indexing uses namespace-specific namespace."""
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

        from vectordb.langchain.namespaces.indexing.pinecone import (
            PineconeNamespaceIndexingPipeline,
        )

        pipeline = PineconeNamespaceIndexingPipeline(
            pinecone_namespace_config, "ns_abc"
        )
        result = pipeline.run()

        assert result["namespace"] == "ns_abc"

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_with_limit_respects_limit(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
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
        pinecone_namespace_config["dataloader"]["limit"] = 2

        from vectordb.langchain.namespaces.indexing.pinecone import (
            PineconeNamespaceIndexingPipeline,
        )

        pipeline = PineconeNamespaceIndexingPipeline(
            pinecone_namespace_config, "ns_123"
        )
        pipeline.run()

        # Verify DataloaderCatalog.create was called with the correct parameters
        mock_get_documents.assert_called_once_with("arc", split="test", limit=2)
