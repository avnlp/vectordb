"""Tests for Pinecone hybrid indexing pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.hybrid_indexing.indexing.pinecone import (
    PineconeHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.search.pinecone import (
    PineconeHybridSearchPipeline,
)


class TestPineconeHybridIndexing:
    """Unit tests for Pinecone hybrid indexing pipeline (indexing and search)."""

    # Indexing tests
    @patch("vectordb.haystack.hybrid_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.pinecone.EmbedderFactory.create_sparse_document_embedder"
    )
    def test_indexing_init_loads_config(
        self,
        mock_sparse_embedder: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = PineconeHybridIndexingPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.index_name == "test-hybrid-index"
        assert pipeline.namespace == "default"

    @patch("vectordb.haystack.hybrid_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.pinecone.EmbedderFactory.create_sparse_document_embedder"
    )
    @patch(
        "vectordb.haystack.hybrid_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_calls_upsert(
        self,
        mock_get_docs: MagicMock,
        mock_sparse_embedder: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents_with_sparse: list[Document],
    ) -> None:
        """Test indexing run method calls upsert."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents_with_sparse
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {
            "documents": sample_documents_with_sparse
        }
        mock_make_embedder.return_value = mock_dense_embedder

        mock_sparse = MagicMock()
        mock_sparse.run.return_value = {"documents": sample_documents_with_sparse}
        mock_sparse_embedder.return_value = mock_sparse

        mock_db = MagicMock()
        mock_db.upsert.return_value = len(sample_documents_with_sparse)
        mock_db_class.return_value = mock_db

        pipeline = PineconeHybridIndexingPipeline(pinecone_config)
        pipeline.run()

        mock_db.upsert.assert_called_once()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            PineconeHybridIndexingPipeline(invalid_config)

    # Search tests
    @patch("vectordb.haystack.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.hybrid_indexing.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch(
        "vectordb.haystack.hybrid_indexing.search.pinecone.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_init_loads_config(
        self,
        mock_sparse_embedder: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = PineconeHybridSearchPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.index_name == "test-hybrid-index"
        assert pipeline.alpha == 0.5

    @patch("vectordb.haystack.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.hybrid_indexing.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch(
        "vectordb.haystack.hybrid_indexing.search.pinecone.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_run_calls_hybrid_search(
        self,
        mock_sparse_embedder: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents_with_sparse: list[Document],
        sample_embedding: list[float],
        sample_sparse_embedding: MagicMock,
    ) -> None:
        """Test search run method calls hybrid_search."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_make_embedder.return_value = mock_embedder

        mock_sparse = MagicMock()
        mock_sparse.run.return_value = {"sparse_embedding": sample_sparse_embedding}
        mock_sparse_embedder.return_value = mock_sparse

        mock_db = MagicMock()
        mock_db.hybrid_search.return_value = sample_documents_with_sparse
        mock_db_class.return_value = mock_db

        pipeline = PineconeHybridSearchPipeline(pinecone_config)
        pipeline.run("test query", top_k=5)

        mock_db.hybrid_search.assert_called_once()

    @patch("vectordb.haystack.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.hybrid_indexing.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch(
        "vectordb.haystack.hybrid_indexing.search.pinecone.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_returns_documents(
        self,
        mock_sparse_embedder: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents_with_sparse: list[Document],
        sample_embedding: list[float],
        sample_sparse_embedding: MagicMock,
    ) -> None:
        """Test search returns documents."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_make_embedder.return_value = mock_embedder

        mock_sparse = MagicMock()
        mock_sparse.run.return_value = {"sparse_embedding": sample_sparse_embedding}
        mock_sparse_embedder.return_value = mock_sparse

        mock_db = MagicMock()
        mock_db.hybrid_search.return_value = sample_documents_with_sparse
        mock_db_class.return_value = mock_db

        pipeline = PineconeHybridSearchPipeline(pinecone_config)
        result = pipeline.run("test query", top_k=5)

        assert "documents" in result
        assert "query" in result
        assert result["db"] == "pinecone"

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            PineconeHybridSearchPipeline(invalid_config)
