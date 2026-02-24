"""Tests for Qdrant sparse indexing pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding

from vectordb.haystack.sparse_indexing.indexing.qdrant import (
    QdrantSparseIndexingPipeline,
)
from vectordb.haystack.sparse_indexing.search.qdrant import (
    QdrantSparseSearchPipeline,
)


class TestQdrantSparseIndexing:
    """Unit tests for Qdrant sparse indexing pipeline."""

    @pytest.fixture
    def qdrant_config(self) -> dict:
        """Create Qdrant-specific test config."""
        return {
            "dataloader": {"name": "triviaqa", "limit": 10},
            "sparse": {
                "model": "prithivida/Splade_PP_en_v1",
            },
            "qdrant": {
                "location": "localhost:6333",
                "collection_name": "test_sparse_collection",
                "api_key": "",
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

    @patch("vectordb.haystack.sparse_indexing.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.qdrant.EmbedderFactory.create_sparse_document_embedder"
    )
    def test_indexing_init_loads_config(
        self,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = QdrantSparseIndexingPipeline(qdrant_config)
        assert pipeline.config == qdrant_config

    @patch("vectordb.haystack.sparse_indexing.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.qdrant.EmbedderFactory.create_sparse_document_embedder"
    )
    @patch("vectordb.haystack.sparse_indexing.indexing.qdrant.DataloaderCatalog.create")
    def test_indexing_run_calls_upsert(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
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
        mock_db.collection_name = "test_sparse_collection"
        mock_db.upsert.return_value = len(sample_documents_with_sparse)
        mock_db_class.return_value = mock_db

        pipeline = QdrantSparseIndexingPipeline(qdrant_config)
        result = pipeline.run()

        mock_db.upsert.assert_called_once()
        assert result["documents_indexed"] == 2

    @patch("vectordb.haystack.sparse_indexing.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.qdrant.EmbedderFactory.create_sparse_document_embedder"
    )
    def test_indexing_create_collection(
        self,
        mock_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test create_collection method."""
        mock_db = MagicMock()
        mock_db.collection_name = "test_sparse_collection"
        mock_db_class.return_value = mock_db

        pipeline = QdrantSparseIndexingPipeline(qdrant_config)
        pipeline.create_collection(dimension=1024, sparse_dimension=30522)

        mock_db.create_collection.assert_called_once()


class TestQdrantSparseSearch:
    """Unit tests for Qdrant sparse search pipeline."""

    @pytest.fixture
    def qdrant_config(self) -> dict:
        """Create Qdrant-specific test config."""
        return {
            "sparse": {
                "model": "prithivida/Splade_PP_en_v1",
            },
            "qdrant": {
                "location": "localhost:6333",
                "collection_name": "test_sparse_collection",
                "api_key": "",
            },
            "query": {"top_k": 5},
        }

    @pytest.fixture
    def sample_sparse_embedding(self) -> SparseEmbedding:
        """Create a sample sparse embedding."""
        return SparseEmbedding(indices=[0, 5, 10, 15], values=[0.5, 0.3, 0.15, 0.05])

    @patch("vectordb.haystack.sparse_indexing.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.qdrant.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_init_loads_config(
        self,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = QdrantSparseSearchPipeline(qdrant_config)
        assert pipeline.config == qdrant_config
        assert pipeline.top_k == 5

    @patch("vectordb.haystack.sparse_indexing.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.qdrant.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_run_calls_search(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
        sample_sparse_embedding: SparseEmbedding,
    ) -> None:
        """Test search run method calls search."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_sparse_embedding}
        mock_make_embedder.return_value = mock_embedder

        mock_result = MagicMock()
        mock_result.payload = {"content": "Test content", "doc_id": "1"}
        mock_result.id = 1
        mock_result.score = 0.9

        mock_db = MagicMock()
        mock_db.search.return_value = [mock_result]
        mock_db_class.return_value = mock_db

        pipeline = QdrantSparseSearchPipeline(qdrant_config)
        result = pipeline.search("test query", top_k=5)

        mock_db.search.assert_called_once()
        assert "documents" in result
        assert "query" in result
        assert len(result["documents"]) == 1

    @patch("vectordb.haystack.sparse_indexing.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.qdrant.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_handles_empty_results(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
        sample_sparse_embedding: SparseEmbedding,
    ) -> None:
        """Test search handles empty results gracefully."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_sparse_embedding}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = QdrantSparseSearchPipeline(qdrant_config)
        result = pipeline.search("test query", top_k=5)

        assert result["documents"] == []

    @patch("vectordb.haystack.sparse_indexing.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.search.qdrant.EmbedderFactory.create_sparse_text_embedder"
    )
    def test_search_handles_none_embedding(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test search handles None embedding gracefully."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": None}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = QdrantSparseSearchPipeline(qdrant_config)
        result = pipeline.search("test query", top_k=5)

        assert result["documents"] == []
        mock_db.search.assert_not_called()
