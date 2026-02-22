"""Tests for Qdrant namespace indexing pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestQdrantNamespaceIndexingPipeline:
    """Unit tests for Qdrant namespace indexing pipeline."""

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_valid_namespace(
        self, mock_embedder, mock_db_cls, qdrant_namespace_config: dict
    ):
        """Test pipeline initialization with valid namespace."""
        mock_embedder.return_value = MagicMock()

        from vectordb.langchain.namespaces.indexing.qdrant import (
            QdrantNamespaceIndexingPipeline,
        )

        pipeline = QdrantNamespaceIndexingPipeline(qdrant_namespace_config, "ns_abc")

        assert pipeline.namespace == "ns_abc"
        assert pipeline.config == qdrant_namespace_config

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_empty_namespace_raises_error(
        self, mock_embedder, mock_db_cls, qdrant_namespace_config: dict
    ):
        """Test initialization with empty namespace raises ValueError."""
        from vectordb.langchain.namespaces.indexing.qdrant import (
            QdrantNamespaceIndexingPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            QdrantNamespaceIndexingPipeline(qdrant_namespace_config, "")

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_run_returns_indexed_count(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        qdrant_namespace_config: dict,
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

        from vectordb.langchain.namespaces.indexing.qdrant import (
            QdrantNamespaceIndexingPipeline,
        )

        pipeline = QdrantNamespaceIndexingPipeline(qdrant_namespace_config, "ns_abc")
        result = pipeline.run()

        assert "documents_indexed" in result
        assert result["namespace"] == "ns_abc"
        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    def test_run_with_no_documents_returns_zero(
        self,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        qdrant_namespace_config: dict,
    ):
        """Test run with no documents returns 0 indexed."""
        mock_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        from vectordb.langchain.namespaces.indexing.qdrant import (
            QdrantNamespaceIndexingPipeline,
        )

        pipeline = QdrantNamespaceIndexingPipeline(qdrant_namespace_config, "ns_abc")
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_run_with_documents_returns_non_zero_count(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_embedder,
        mock_db_cls,
        qdrant_namespace_config: dict,
    ):
        """Test run indexes non-empty documents and returns indexed count."""
        mock_embedder.return_value = MagicMock()

        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (documents, [[0.1] * 384, [0.2] * 384])

        mock_db_instance = MagicMock()
        mock_db_instance.upsert.return_value = 2
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.indexing.qdrant import (
            QdrantNamespaceIndexingPipeline,
        )

        pipeline = QdrantNamespaceIndexingPipeline(qdrant_namespace_config, "ns_abc")
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        assert result["namespace"] == "ns_abc"
        mock_embed_documents.assert_called_once_with(pipeline.embedder, documents)
        mock_db_instance.upsert.assert_called_once()
