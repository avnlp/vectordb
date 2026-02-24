"""Unit tests for Qdrant hybrid indexing pipelines."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.hybrid_indexing.indexing.qdrant import (
    QdrantHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.search.qdrant import QdrantHybridSearchPipeline


class TestQdrantHybridIndexing:
    """Test class for Qdrant hybrid indexing feature."""

    @patch("vectordb.haystack.hybrid_indexing.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.qdrant.EmbedderFactory")
    def test_indexing_init_loads_config(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        qdrant_config: dict[str, Any],
    ) -> None:
        """Test that indexing pipeline initializes from config."""
        mock_embedder = MagicMock()
        mock_embedder_factory.create_document_embedder.return_value = mock_embedder
        mock_embedder_factory.create_sparse_document_embedder.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()

        pipeline = QdrantHybridIndexingPipeline(qdrant_config)

        assert pipeline.config == qdrant_config
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.dimension == 384
        mock_embedder_factory.create_document_embedder.assert_called_once()

    @patch("vectordb.haystack.hybrid_indexing.indexing.qdrant.DataloaderCatalog.create")
    @patch("vectordb.haystack.hybrid_indexing.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.qdrant.EmbedderFactory")
    def test_indexing_run_calls_index_documents(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        mock_get_docs: MagicMock,
        qdrant_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test that run() loads, embeds, and indexes documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_dense_embedder = MagicMock()
        embedded_docs = [
            Document(content=doc.content, embedding=[0.1] * 384)
            for doc in sample_documents
        ]
        mock_dense_embedder.run.return_value = {"documents": embedded_docs}
        mock_embedder_factory.create_document_embedder.return_value = (
            mock_dense_embedder
        )
        mock_embedder_factory.create_sparse_document_embedder.return_value = MagicMock(
            run=MagicMock(return_value={"documents": embedded_docs})
        )

        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        pipeline = QdrantHybridIndexingPipeline(qdrant_config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        assert result["db"] == "qdrant"
        assert result["collection_name"] == "test_hybrid"
        mock_loader.load.assert_called_once()
        mock_db.create_collection.assert_called_once()
        mock_db.index_documents.assert_called()

    @patch("vectordb.haystack.hybrid_indexing.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.qdrant.EmbedderFactory")
    def test_indexing_invalid_config_raises(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
    ) -> None:
        """Test that invalid config raises ValueError."""
        invalid_config: dict[str, Any] = {
            "embeddings": {"model": "test-model"},
        }

        with pytest.raises(ValueError):
            QdrantHybridIndexingPipeline(invalid_config)

    @patch("vectordb.haystack.hybrid_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.qdrant.EmbedderFactory")
    def test_search_init_loads_config(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        qdrant_config: dict[str, Any],
    ) -> None:
        """Test that search pipeline initializes from config."""
        mock_embedder = MagicMock()
        mock_embedder_factory.create_text_embedder.return_value = mock_embedder
        mock_embedder_factory.create_sparse_text_embedder.return_value = MagicMock()
        mock_db_cls.return_value = MagicMock()

        pipeline = QdrantHybridSearchPipeline(qdrant_config)

        assert pipeline.config == qdrant_config
        assert pipeline.collection_name == "test_hybrid"
        mock_embedder_factory.create_text_embedder.assert_called_once()

    @patch("vectordb.haystack.hybrid_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.qdrant.EmbedderFactory")
    def test_search_run_calls_search(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        qdrant_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test that run() embeds query and searches."""
        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.create_text_embedder.return_value = mock_dense_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {"sparse_embedding": MagicMock()}
        mock_embedder_factory.create_sparse_text_embedder.return_value = (
            mock_sparse_embedder
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_cls.return_value = mock_db

        pipeline = QdrantHybridSearchPipeline(qdrant_config)
        result = pipeline.run("What is machine learning?", top_k=5)

        assert result["documents"] == sample_documents
        assert result["query"] == "What is machine learning?"
        assert result["db"] == "qdrant"
        mock_dense_embedder.run.assert_called_once()
        mock_db.search.assert_called_once()

    @patch("vectordb.haystack.hybrid_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.qdrant.EmbedderFactory")
    def test_search_returns_documents(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        qdrant_config: dict[str, Any],
        sample_documents_with_sparse: list[Document],
    ) -> None:
        """Test that search returns documents with embeddings."""
        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.create_text_embedder.return_value = mock_dense_embedder
        mock_embedder_factory.create_sparse_text_embedder.return_value = MagicMock(
            run=MagicMock(return_value={"sparse_embedding": MagicMock()})
        )

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents_with_sparse
        mock_db_cls.return_value = mock_db

        pipeline = QdrantHybridSearchPipeline(qdrant_config)
        result = pipeline.run("deep learning", top_k=3)

        assert len(result["documents"]) == 3
        for doc in result["documents"]:
            assert doc.embedding is not None
            assert doc.sparse_embedding is not None
