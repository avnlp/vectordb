"""Tests for Weaviate BM25 indexing pipeline (indexing and search).

Note: Weaviate computes BM25 internally from stored text at query time,
so no external sparse embeddings are needed.
"""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.sparse_indexing.indexing.weaviate import (
    WeaviateBM25IndexingPipeline,
)
from vectordb.haystack.sparse_indexing.search.weaviate import (
    WeaviateBM25SearchPipeline,
)


class TestWeaviateBM25Indexing:
    """Unit tests for Weaviate BM25 indexing pipeline."""

    @pytest.fixture
    def weaviate_config(self) -> dict:
        """Create Weaviate-specific test config."""
        return {
            "dataloader": {"name": "triviaqa", "limit": 10},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "index_name": "TestBM25",
            },
            "indexing": {"batch_size": 100},
            "query": {"top_k": 5},
        }

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create sample documents."""
        return [
            Document(
                content="Machine learning is a subset of artificial intelligence.",
                meta={"id": "1", "source": "wiki"},
            ),
            Document(
                content="Deep learning uses neural networks with multiple layers.",
                meta={"id": "2", "source": "paper"},
            ),
        ]

    @patch("vectordb.haystack.sparse_indexing.indexing.weaviate.WeaviateVectorDB")
    def test_indexing_init_loads_config(
        self,
        mock_db: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = WeaviateBM25IndexingPipeline(weaviate_config)
        assert pipeline.config == weaviate_config
        assert pipeline.batch_size == 100

    @patch("vectordb.haystack.sparse_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.haystack.sparse_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_run_calls_upsert(
        self,
        mock_get_docs: MagicMock,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test indexing run method calls upsert."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_db = MagicMock()
        mock_db.upsert.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = WeaviateBM25IndexingPipeline(weaviate_config)
        result = pipeline.run()

        mock_db.upsert.assert_called_once()
        assert result["documents_indexed"] == 2

    @patch("vectordb.haystack.sparse_indexing.indexing.weaviate.WeaviateVectorDB")
    def test_indexing_create_collection(
        self,
        mock_db_class: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test create_collection method."""
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = WeaviateBM25IndexingPipeline(weaviate_config)
        pipeline.create_collection()

        mock_db.create_collection.assert_called_once()


class TestWeaviateBM25Search:
    """Unit tests for Weaviate BM25 search pipeline."""

    @pytest.fixture
    def weaviate_config(self) -> dict:
        """Create Weaviate-specific test config."""
        return {
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "index_name": "TestBM25",
            },
            "query": {"top_k": 5},
        }

    @pytest.fixture
    def mock_search_results(self) -> list[MagicMock]:
        """Create mock search results."""
        result1 = MagicMock()
        result1.content = "Test content 1"
        result1.id = "1"
        result1.score = 0.9
        result1.meta = {"source": "wiki"}

        result2 = MagicMock()
        result2.content = "Test content 2"
        result2.id = "2"
        result2.score = 0.8
        result2.meta = {"source": "paper"}

        return [result1, result2]

    @patch("vectordb.haystack.sparse_indexing.search.weaviate.WeaviateVectorDB")
    def test_search_init_loads_config(
        self,
        mock_db: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = WeaviateBM25SearchPipeline(weaviate_config)
        assert pipeline.config == weaviate_config
        assert pipeline.top_k == 5

    @patch("vectordb.haystack.sparse_indexing.search.weaviate.WeaviateVectorDB")
    def test_search_run_calls_bm25_search(
        self,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        mock_search_results: list[MagicMock],
    ) -> None:
        """Test search run method calls bm25_search."""
        mock_db = MagicMock()
        mock_db.bm25_search.return_value = mock_search_results
        mock_db_class.return_value = mock_db

        pipeline = WeaviateBM25SearchPipeline(weaviate_config)
        result = pipeline.search("test query", top_k=5)

        mock_db.bm25_search.assert_called_once_with(query="test query", top_k=5)
        assert "documents" in result
        assert "query" in result
        assert len(result["documents"]) == 2

    @patch("vectordb.haystack.sparse_indexing.search.weaviate.WeaviateVectorDB")
    def test_search_handles_empty_results(
        self,
        mock_db_class: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test search handles empty results gracefully."""
        mock_db = MagicMock()
        mock_db.bm25_search.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = WeaviateBM25SearchPipeline(weaviate_config)
        result = pipeline.search("test query", top_k=5)

        assert result["documents"] == []
