"""Unit tests for Chroma hybrid indexing and search pipelines."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document


class TestChromaHybridIndexing:
    """Tests for ChromaHybridIndexingPipeline."""

    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    def test_init_loads_config(
        self,
        mock_db_class: MagicMock,
        mock_embedder_factory: MagicMock,
        chroma_config: dict[str, Any],
    ) -> None:
        """Test that __init__ properly loads config and initializes components."""
        from vectordb.haystack.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
        )

        mock_dense_embedder = MagicMock()
        mock_sparse_embedder = MagicMock()
        mock_embedder_factory.create_document_embedder.return_value = (
            mock_dense_embedder
        )
        mock_embedder_factory.create_sparse_document_embedder.return_value = (
            mock_sparse_embedder
        )

        pipeline = ChromaHybridIndexingPipeline(chroma_config)

        assert pipeline.config == chroma_config
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.dense_embedder == mock_dense_embedder
        assert pipeline.sparse_embedder == mock_sparse_embedder
        mock_db_class.assert_called_once()
        mock_embedder_factory.create_document_embedder.assert_called_once_with(
            chroma_config
        )

    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.DataloaderCatalog.create")
    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    def test_run_calls_upsert(
        self,
        mock_db_class: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_get_docs: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
        sample_documents_with_sparse: list[Document],
    ) -> None:
        """Test that run() loads docs, embeds them, and calls upsert."""
        from vectordb.haystack.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
        )

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {"documents": sample_documents}
        mock_embedder_factory.create_document_embedder.return_value = (
            mock_dense_embedder
        )

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {
            "documents": sample_documents_with_sparse
        }
        mock_embedder_factory.create_sparse_document_embedder.return_value = (
            mock_sparse_embedder
        )

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        pipeline = ChromaHybridIndexingPipeline(chroma_config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents_with_sparse)
        assert result["db"] == "chroma"
        assert result["collection_name"] == "test_hybrid"
        mock_db.create_collection.assert_called_once()
        mock_db.upsert.assert_called()

    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.DataloaderCatalog.create")
    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    def test_run_with_empty_documents(
        self,
        mock_db_class: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_get_docs: MagicMock,
        chroma_config: dict[str, Any],
    ) -> None:
        """Test run() returns early when no documents are loaded."""
        from vectordb.haystack.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
        )

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        pipeline = ChromaHybridIndexingPipeline(chroma_config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["db"] == "chroma"
        mock_db.upsert.assert_not_called()

    def test_invalid_config_raises(self) -> None:
        """Test that missing chroma config raises ValueError."""
        from vectordb.haystack.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
        )

        invalid_config: dict[str, Any] = {
            "dataloader": {"type": "arc"},
            "embeddings": {"model": "test-model"},
        }

        with pytest.raises(ValueError):
            ChromaHybridIndexingPipeline(invalid_config)


class TestChromaHybridSearch:
    """Tests for ChromaHybridSearchPipeline."""

    @patch("vectordb.haystack.hybrid_indexing.search.chroma.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.search.chroma.ChromaVectorDB")
    def test_init_loads_config(
        self,
        mock_db_class: MagicMock,
        mock_embedder_factory: MagicMock,
        chroma_config: dict[str, Any],
    ) -> None:
        """Test that __init__ properly loads config and initializes components."""
        from vectordb.haystack.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        mock_text_embedder = MagicMock()
        mock_sparse_text_embedder = MagicMock()
        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder
        mock_embedder_factory.create_sparse_text_embedder.return_value = (
            mock_sparse_text_embedder
        )

        pipeline = ChromaHybridSearchPipeline(chroma_config)

        assert pipeline.config == chroma_config
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.dense_embedder == mock_text_embedder
        assert pipeline.sparse_embedder == mock_sparse_text_embedder
        mock_db_class.assert_called_once()
        mock_embedder_factory.create_text_embedder.assert_called_once_with(
            chroma_config
        )

    @patch("vectordb.haystack.hybrid_indexing.search.chroma.ResultMerger")
    @patch("vectordb.haystack.hybrid_indexing.search.chroma.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.search.chroma.ChromaVectorDB")
    def test_run_calls_search_methods(
        self,
        mock_db_class: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_result_merger: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
        sample_embedding: list[float],
        sample_sparse_embedding: Any,
    ) -> None:
        """Test that run() calls search methods and returns fused documents."""
        from vectordb.haystack.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_documents

        mock_text_embedder = MagicMock()
        mock_text_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {
            "sparse_embedding": sample_sparse_embedding
        }
        mock_embedder_factory.create_sparse_text_embedder.return_value = (
            mock_sparse_embedder
        )

        mock_result_merger.fuse.return_value = sample_documents[:2]

        pipeline = ChromaHybridSearchPipeline(chroma_config)
        result = pipeline.run(query="test query", top_k=5)

        assert "documents" in result
        assert result["query"] == "test query"
        assert result["db"] == "chroma"
        assert mock_db.search.call_count == 2  # dense + sparse
        mock_result_merger.fuse.assert_called_once()

    @patch("vectordb.haystack.hybrid_indexing.search.chroma.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.search.chroma.ChromaVectorDB")
    def test_run_returns_documents(
        self,
        mock_db_class: MagicMock,
        mock_embedder_factory: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test that run() returns documents in expected format."""
        from vectordb.haystack.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        config_no_sparse = {k: v for k, v in chroma_config.items() if k != "sparse"}

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.search.return_value = sample_documents

        mock_text_embedder = MagicMock()
        mock_text_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder
        mock_embedder_factory.create_sparse_text_embedder.return_value = None

        pipeline = ChromaHybridSearchPipeline(config_no_sparse)
        result = pipeline.run(query="what is ML?", top_k=3)

        assert "documents" in result
        assert len(result["documents"]) <= 3
        assert result["query"] == "what is ML?"
        mock_db.search.assert_called_once()

    def test_invalid_config_raises(self) -> None:
        """Test that missing chroma config raises ValueError."""
        from vectordb.haystack.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        invalid_config: dict[str, Any] = {
            "embeddings": {"model": "test-model"},
        }

        with pytest.raises(ValueError):
            ChromaHybridSearchPipeline(invalid_config)
