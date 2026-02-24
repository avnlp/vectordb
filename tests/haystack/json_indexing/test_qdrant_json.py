"""Unit tests for Qdrant JSON indexing pipelines."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.json_indexing import QdrantJSONIndexer, QdrantJSONSearcher


class TestQdrantJSONIndexing:
    """Test class for Qdrant JSON indexing feature."""

    @pytest.fixture
    def qdrant_config(self) -> dict[str, Any]:
        """Return a sample Qdrant configuration."""
        return {
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_collection",
            },
            "collection": {"name": "test_collection"},
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"type": "triviaqa", "split": "test", "limit": 10},
            "search": {"top_k": 10},
        }

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Return sample Haystack documents."""
        return [
            Document(content="What is machine learning?", meta={"category": "AI"}),
            Document(content="How does deep learning work?", meta={"category": "AI"}),
            Document(content="What is Python?", meta={"category": "programming"}),
        ]

    @patch("vectordb.haystack.json_indexing.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.json_indexing.indexing.qdrant.create_document_embedder")
    @patch("vectordb.haystack.json_indexing.indexing.qdrant.get_embedding_dimension")
    @patch("vectordb.haystack.json_indexing.indexing.qdrant.DataloaderCatalog")
    def test_indexing_unit(
        self,
        mock_registry: MagicMock,
        mock_get_dim: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        qdrant_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test Qdrant indexing pipeline with mocked dependencies."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.return_value = mock_embedder
        mock_get_dim.return_value = 384

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_registry.create.return_value = mock_loader

        indexer = QdrantJSONIndexer(qdrant_config)
        result = indexer.run()

        mock_db_cls.assert_called_once_with(config=qdrant_config)
        mock_db.create_collection.assert_called_once()
        mock_db.index_documents.assert_called_once()
        assert result["documents_indexed"] == 3

    @patch("vectordb.haystack.json_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.json_indexing.search.qdrant.create_text_embedder")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        qdrant_config: dict[str, Any],
    ) -> None:
        """Test Qdrant search pipeline with mocked dependencies."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.search.return_value = [
            {"content": "result 1", "score": 0.9},
            {"content": "result 2", "score": 0.8},
        ]

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        searcher = QdrantJSONSearcher(qdrant_config)
        results = searcher.search("What is AI?", top_k=5)

        mock_db_cls.assert_called_once_with(config=qdrant_config)
        mock_embedder.run.assert_called_once_with(text="What is AI?")
        mock_db.search.assert_called_once()
        assert len(results) == 2

    @patch("vectordb.haystack.json_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.json_indexing.search.qdrant.create_text_embedder")
    def test_search_with_filters_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        qdrant_config: dict[str, Any],
    ) -> None:
        """Test Qdrant search with filters."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.search.return_value = [{"content": "AI result", "score": 0.95}]

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        searcher = QdrantJSONSearcher(qdrant_config)
        results = searcher.search(
            "machine learning", filters={"category": "AI"}, top_k=10
        )

        mock_db.search.assert_called_once()
        call_kwargs = mock_db.search.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["query_filter"] is not None
        assert len(results) == 1

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("QDRANT_API_KEY"),
        reason="Requires running Qdrant instance - run with --integration flag",
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Qdrant indexing (requires running Qdrant)."""
        pytest.skip("Requires running Qdrant instance - run with --integration flag")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("QDRANT_API_KEY"),
        reason="Requires running Qdrant instance - run with --integration flag",
    )
    def test_search_integration(self) -> None:
        """Integration test for Qdrant search (requires running Qdrant)."""
        pytest.skip("Requires running Qdrant instance - run with --integration flag")
