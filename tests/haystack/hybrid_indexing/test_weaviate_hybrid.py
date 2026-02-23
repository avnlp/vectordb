"""Unit tests for Weaviate hybrid indexing and search pipelines."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.hybrid_indexing.indexing.weaviate import (
    WeaviateHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.search.weaviate import (
    WeaviateHybridSearchPipeline,
)


class TestWeaviateHybridIndexing:
    """Tests for WeaviateHybridIndexingPipeline."""

    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.EmbedderFactory")
    def test_init_loads_config(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        weaviate_config: dict[str, Any],
    ) -> None:
        """Test that init loads configuration correctly."""
        mock_embedder_factory.create_document_embedder.return_value = MagicMock()
        mock_embedder_factory.create_sparse_document_embedder.return_value = MagicMock()

        pipeline = WeaviateHybridIndexingPipeline(weaviate_config)

        assert pipeline.collection_name == "TestHybrid"
        assert pipeline.dimension == 384
        mock_weaviate_db.assert_called_once()
        mock_embedder_factory.create_document_embedder.assert_called_once_with(
            weaviate_config
        )

    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.EmbedderFactory")
    def test_run_calls_upsert(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        mock_get_docs: MagicMock,
        weaviate_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test that run method calls upsert with embedded documents."""
        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {"documents": sample_documents}
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {"documents": sample_documents}
        mock_embedder_factory.create_document_embedder.return_value = (
            mock_dense_embedder
        )
        mock_embedder_factory.create_sparse_document_embedder.return_value = (
            mock_sparse_embedder
        )

        mock_db_instance = MagicMock()
        mock_weaviate_db.return_value = mock_db_instance

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        pipeline = WeaviateHybridIndexingPipeline(weaviate_config)
        result = pipeline.run()

        mock_db_instance.create_collection.assert_called_once_with(
            collection_name="TestHybrid",
            dimension=384,
        )
        mock_db_instance.upsert.assert_called()
        assert result["documents_indexed"] == 3
        assert result["db"] == "weaviate"
        assert result["collection_name"] == "TestHybrid"

    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.EmbedderFactory")
    def test_invalid_config_raises(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
    ) -> None:
        """Test that invalid config raises ValueError."""
        invalid_config: dict[str, Any] = {"embeddings": {"model": "test"}}

        with pytest.raises(ValueError):
            WeaviateHybridIndexingPipeline(invalid_config)

    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.EmbedderFactory")
    def test_run_with_sparse_embedder(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        mock_get_docs: MagicMock,
        weaviate_config: dict[str, Any],
        sample_documents_with_sparse: list[Document],
    ) -> None:
        """Test that run uses sparse embedder when configured."""
        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {
            "documents": sample_documents_with_sparse
        }
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {
            "documents": sample_documents_with_sparse
        }

        mock_embedder_factory.create_document_embedder.return_value = (
            mock_dense_embedder
        )
        mock_embedder_factory.create_sparse_document_embedder.return_value = (
            mock_sparse_embedder
        )

        mock_db_instance = MagicMock()
        mock_weaviate_db.return_value = mock_db_instance

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents_with_sparse
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        pipeline = WeaviateHybridIndexingPipeline(weaviate_config)
        result = pipeline.run()

        mock_dense_embedder.run.assert_called_once()
        mock_sparse_embedder.run.assert_called_once()
        assert result["documents_indexed"] == 3

    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.indexing.weaviate.EmbedderFactory")
    def test_run_with_no_documents(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        mock_get_docs: MagicMock,
        weaviate_config: dict[str, Any],
    ) -> None:
        """Test that run handles empty document list."""
        mock_embedder_factory.create_document_embedder.return_value = MagicMock()
        mock_embedder_factory.create_sparse_document_embedder.return_value = MagicMock()
        mock_db_instance = MagicMock()
        mock_weaviate_db.return_value = mock_db_instance

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        pipeline = WeaviateHybridIndexingPipeline(weaviate_config)
        result = pipeline.run()

        mock_db_instance.upsert.assert_not_called()
        assert result["documents_indexed"] == 0


class TestWeaviateHybridSearch:
    """Tests for WeaviateHybridSearchPipeline."""

    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.EmbedderFactory")
    def test_init_loads_config(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        weaviate_config: dict[str, Any],
    ) -> None:
        """Test that init loads configuration correctly."""
        mock_embedder_factory.create_text_embedder.return_value = MagicMock()

        pipeline = WeaviateHybridSearchPipeline(weaviate_config)

        assert pipeline.collection_name == "TestHybrid"
        assert pipeline.alpha == 0.5
        mock_weaviate_db.assert_called_once()
        mock_embedder_factory.create_text_embedder.assert_called_once_with(
            weaviate_config
        )

    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.EmbedderFactory")
    def test_run_calls_hybrid_search(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        weaviate_config: dict[str, Any],
        sample_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test that run method calls hybrid_search with correct parameters."""
        mock_text_embedder = MagicMock()
        mock_text_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder

        mock_db_instance = MagicMock()
        mock_db_instance.hybrid_search.return_value = sample_documents
        mock_weaviate_db.return_value = mock_db_instance

        pipeline = WeaviateHybridSearchPipeline(weaviate_config)
        result = pipeline.run(query="machine learning", top_k=5)

        mock_text_embedder.run.assert_called_once_with(text="machine learning")
        mock_db_instance.hybrid_search.assert_called_once_with(
            query="machine learning",
            query_embedding=sample_embedding,
            collection_name="TestHybrid",
            top_k=5,
            alpha=0.5,
            filter=None,
        )
        assert result["documents"] == sample_documents
        assert result["query"] == "machine learning"
        assert result["db"] == "weaviate"

    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.EmbedderFactory")
    def test_run_with_filters(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        weaviate_config: dict[str, Any],
        sample_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test that run method passes filters to hybrid_search."""
        mock_text_embedder = MagicMock()
        mock_text_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder

        mock_db_instance = MagicMock()
        mock_db_instance.hybrid_search.return_value = [sample_documents[0]]
        mock_weaviate_db.return_value = mock_db_instance

        pipeline = WeaviateHybridSearchPipeline(weaviate_config)
        filters = {"source": "wiki"}
        result = pipeline.run(query="AI", top_k=3, filters=filters)

        mock_db_instance.hybrid_search.assert_called_once_with(
            query="AI",
            query_embedding=sample_embedding,
            collection_name="TestHybrid",
            top_k=3,
            alpha=0.5,
            filter=filters,
        )
        assert len(result["documents"]) == 1

    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.EmbedderFactory")
    def test_invalid_config_raises(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
    ) -> None:
        """Test that invalid config raises ValueError."""
        invalid_config: dict[str, Any] = {"embeddings": {"model": "test"}}

        with pytest.raises(ValueError):
            WeaviateHybridSearchPipeline(invalid_config)

    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.hybrid_indexing.search.weaviate.EmbedderFactory")
    def test_run_returns_empty_documents(
        self,
        mock_embedder_factory: MagicMock,
        mock_weaviate_db: MagicMock,
        weaviate_config: dict[str, Any],
        sample_embedding: list[float],
    ) -> None:
        """Test that run handles empty search results."""
        mock_text_embedder = MagicMock()
        mock_text_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder

        mock_db_instance = MagicMock()
        mock_db_instance.hybrid_search.return_value = []
        mock_weaviate_db.return_value = mock_db_instance

        pipeline = WeaviateHybridSearchPipeline(weaviate_config)
        result = pipeline.run(query="nonexistent topic", top_k=5)

        assert result["documents"] == []
        assert result["query"] == "nonexistent topic"
