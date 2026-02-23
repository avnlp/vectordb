"""Unit tests for Weaviate JSON indexing pipelines."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.json_indexing import WeaviateJSONIndexer, WeaviateJSONSearcher


class TestWeaviateJSONIndexing:
    """Test class for Weaviate JSON indexing feature."""

    @pytest.fixture
    def weaviate_config(self) -> dict[str, Any]:
        """Return a sample Weaviate configuration."""
        return {
            "weaviate": {
                "cluster_url": "http://localhost:8080",
                "api_key": "test-api-key",
            },
            "collection": {"name": "TestCollection"},
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

    @patch("vectordb.haystack.json_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.json_indexing.indexing.weaviate.create_document_embedder")
    @patch("vectordb.haystack.json_indexing.indexing.weaviate.DataloaderCatalog")
    def test_indexing_unit(
        self,
        mock_registry: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test Weaviate indexing pipeline with mocked dependencies."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.return_value = mock_embedder

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_registry.create.return_value = mock_loader

        indexer = WeaviateJSONIndexer(weaviate_config)
        result = indexer.run()

        mock_db_cls.assert_called_once_with(
            cluster_url="http://localhost:8080", api_key="test-api-key"
        )
        mock_db.create_collection.assert_called_once()
        mock_db.upsert_documents.assert_called_once()
        assert result["documents_indexed"] == 3

    @patch("vectordb.haystack.json_indexing.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.json_indexing.search.weaviate.create_text_embedder")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_config: dict[str, Any],
    ) -> None:
        """Test Weaviate search pipeline with mocked dependencies."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.hybrid_search.return_value = [
            {"content": "result 1", "score": 0.9},
            {"content": "result 2", "score": 0.8},
        ]

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        searcher = WeaviateJSONSearcher(weaviate_config)
        results = searcher.search("What is AI?", top_k=5)

        mock_db_cls.assert_called_once_with(
            cluster_url="http://localhost:8080", api_key="test-api-key"
        )
        mock_embedder.run.assert_called_once_with(text="What is AI?")
        mock_db.hybrid_search.assert_called_once()
        assert len(results) == 2

    @patch("vectordb.haystack.json_indexing.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.json_indexing.search.weaviate.create_text_embedder")
    def test_search_with_filters_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_config: dict[str, Any],
    ) -> None:
        """Test Weaviate search with filters."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.hybrid_search.return_value = [{"content": "AI result", "score": 0.95}]

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        searcher = WeaviateJSONSearcher(weaviate_config)
        results = searcher.search(
            "machine learning", filters={"category": "AI"}, top_k=10
        )

        mock_db.hybrid_search.assert_called_once()
        call_kwargs = mock_db.hybrid_search.call_args[1]
        assert call_kwargs["limit"] == 10
        assert len(results) == 1

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("WEAVIATE_API_KEY"),
        reason="Requires running Weaviate instance - run with --integration flag",
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Weaviate indexing (requires running Weaviate)."""
        pytest.skip("Requires running Weaviate instance - run with --integration flag")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("WEAVIATE_API_KEY"),
        reason="Requires running Weaviate instance - run with --integration flag",
    )
    def test_search_integration(self) -> None:
        """Integration test for Weaviate search (requires running Weaviate)."""
        pytest.skip("Requires running Weaviate instance - run with --integration flag")
