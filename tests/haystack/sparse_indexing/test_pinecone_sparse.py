"""Tests for Pinecone sparse indexing pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding

from vectordb.haystack.sparse_indexing.indexing.pinecone import (
    PineconeSparseIndexingPipeline,
)
from vectordb.haystack.sparse_indexing.search.pinecone import (
    PineconeSparseSearchPipeline,
)


class TestPineconeSparseIndexing:
    """Unit tests for Pinecone sparse indexing pipeline."""

    @pytest.fixture
    def pinecone_config(self) -> dict:
        """Create Pinecone-specific test config."""
        return {
            "dataloader": {"name": "triviaqa", "limit": 10},
            "sparse": {
                "model": "prithivida/Splade_PP_en_v1",
            },
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-sparse-index",
                "namespace": "default",
            },
            "indexing": {"batch_size": 100},
            "query": {"top_k": 5},
        }

    @pytest.fixture
    def sample_documents_with_sparse(self) -> list[Document]:
        """Create sample documents with sparse embeddings."""
        return [
            Document(
                content="Machine learning is a subset of artificial intelligence.",
                meta={"id": "1", "source": "wiki"},
                sparse_embedding=SparseEmbedding(
                    indices=[0, 5, 10], values=[0.5, 0.3, 0.2]
                ),
            ),
            Document(
                content="Deep learning uses neural networks with multiple layers.",
                meta={"id": "2", "source": "paper"},
                sparse_embedding=SparseEmbedding(
                    indices=[1, 6, 11], values=[0.4, 0.35, 0.25]
                ),
            ),
        ]

    @pytest.fixture
    def sample_sparse_embedding(self) -> SparseEmbedding:
        """Create a sample sparse embedding."""
        return SparseEmbedding(indices=[0, 5, 10, 15], values=[0.5, 0.3, 0.15, 0.05])

    @patch("vectordb.haystack.sparse_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.pinecone.EmbedderFactory.create_sparse_document_embedder"
    )
    def test_indexing_init_loads_config(
        self,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = PineconeSparseIndexingPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.namespace == "default"

    @patch("vectordb.haystack.sparse_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.pinecone.EmbedderFactory.create_sparse_document_embedder"
    )
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_calls_upsert(
        self,
        mock_get_docs: MagicMock,
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

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {
            "documents": sample_documents_with_sparse
        }
        mock_make_embedder.return_value = mock_sparse_embedder

        mock_db = MagicMock()
        mock_db.upsert.return_value = len(sample_documents_with_sparse)
        mock_db_class.return_value = mock_db

        pipeline = PineconeSparseIndexingPipeline(pinecone_config)
        result = pipeline.run()

        mock_db.upsert.assert_called_once()
        assert result["documents_indexed"] == 2

    @patch("vectordb.haystack.sparse_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.pinecone.EmbedderFactory.create_sparse_document_embedder"
    )
    def test_indexing_create_index(
        self,
        mock_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test create_index method."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = PineconeSparseIndexingPipeline(pinecone_config)
        pipeline.create_index(dimension=1, metric="dotproduct")

        mock_db.create_index.assert_called_once_with(dimension=1, metric="dotproduct")


class TestPineconeSparseSearch:
    """Unit tests for Pinecone sparse search pipeline."""

    @pytest.fixture
    def pinecone_config(self) -> dict:
        """Create Pinecone-specific test config."""
        return {
            "sparse": {
                "model": "prithivida/Splade_PP_en_v1",
            },
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-sparse-index",
                "namespace": "default",
            },
            "query": {"top_k": 5},
        }

    @pytest.fixture
    def sample_sparse_embedding(self) -> SparseEmbedding:
        """Create a sample sparse embedding."""
        return SparseEmbedding(indices=[0, 5, 10, 15], values=[0.5, 0.3, 0.15, 0.05])

    @patch("vectordb.haystack.sparse_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.pinecone.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_init_loads_config(
        self,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = PineconeSparseSearchPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.top_k == 5
        assert pipeline.namespace == "default"

    @patch("vectordb.haystack.sparse_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.pinecone.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_run_calls_query(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_sparse_embedding: SparseEmbedding,
    ) -> None:
        """Test search run method calls query."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_sparse_embedding}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = {
            "matches": [
                {"id": "1", "score": 0.9, "metadata": {"content": "Test content 1"}},
                {"id": "2", "score": 0.8, "metadata": {"content": "Test content 2"}},
            ]
        }
        mock_db_class.return_value = mock_db

        pipeline = PineconeSparseSearchPipeline(pinecone_config)
        result = pipeline.search("test query", top_k=5)

        mock_db.query.assert_called_once()
        assert "documents" in result
        assert "query" in result
        assert len(result["documents"]) == 2

    @patch("vectordb.haystack.sparse_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.pinecone.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_handles_empty_results(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_sparse_embedding: SparseEmbedding,
    ) -> None:
        """Test search handles empty results gracefully."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_sparse_embedding}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = {"matches": []}
        mock_db_class.return_value = mock_db

        pipeline = PineconeSparseSearchPipeline(pinecone_config)
        result = pipeline.search("test query", top_k=5)

        assert result["documents"] == []

    @patch("vectordb.haystack.sparse_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.pinecone.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_handles_none_embedding(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test search handles None embedding gracefully."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": None}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = PineconeSparseSearchPipeline(pinecone_config)
        result = pipeline.search("test query", top_k=5)

        assert result["documents"] == []
        mock_db.query.assert_not_called()
