"""Unit tests for Chroma JSON indexing pipelines."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.json_indexing import ChromaJSONIndexer, ChromaJSONSearcher


class TestChromaJSONIndexing:
    """Test class for Chroma JSON indexing feature."""

    @pytest.fixture
    def chroma_config(self) -> dict[str, Any]:
        """Return a sample Chroma configuration."""
        return {
            "chroma": {
                "persist_directory": "/tmp/chroma_test",
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

    @patch("vectordb.haystack.json_indexing.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.json_indexing.indexing.chroma.create_document_embedder")
    @patch("vectordb.haystack.json_indexing.indexing.chroma.DataloaderCatalog")
    def test_indexing_unit(
        self,
        mock_registry: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        chroma_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test Chroma indexing pipeline with mocked dependencies."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_embedder = MagicMock()
        mock_embedder_factory.return_value = mock_embedder

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_registry.create.return_value = mock_loader

        indexer = ChromaJSONIndexer(chroma_config)
        result = indexer.run()

        mock_db_cls.assert_called_once_with(config=chroma_config)
        mock_db.create_collection.assert_called_once()
        mock_db.upsert_documents.assert_called_once()
        assert result["documents_indexed"] == 3

    @patch("vectordb.haystack.json_indexing.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.json_indexing.search.chroma.create_text_embedder")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        chroma_config: dict[str, Any],
    ) -> None:
        """Test Chroma search pipeline with mocked dependencies."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.search.return_value = [
            {"content": "result 1", "score": 0.9},
            {"content": "result 2", "score": 0.8},
        ]

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        searcher = ChromaJSONSearcher(chroma_config)
        results = searcher.search("What is AI?", top_k=5)

        mock_db_cls.assert_called_once_with(config=chroma_config)
        mock_embedder.run.assert_called_once_with(text="What is AI?")
        mock_db.search.assert_called_once()
        assert len(results) == 2

    @patch("vectordb.haystack.json_indexing.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.json_indexing.search.chroma.create_text_embedder")
    def test_search_with_filters_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        chroma_config: dict[str, Any],
    ) -> None:
        """Test Chroma search with filters."""
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_db.search.return_value = [{"content": "AI result", "score": 0.95}]

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        searcher = ChromaJSONSearcher(chroma_config)
        results = searcher.search(
            "machine learning", filters={"category": "AI"}, top_k=10
        )

        mock_db.search.assert_called_once()
        call_kwargs = mock_db.search.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["where"] == {"category": {"$eq": "AI"}}
        assert len(results) == 1

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("CHROMA_PERSIST_DIRECTORY"),
        reason="Requires running Chroma instance - run with --integration flag",
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Chroma indexing."""
        pytest.skip("Requires running Chroma instance - run with --integration flag")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("CHROMA_PERSIST_DIRECTORY"),
        reason="Requires running Chroma instance - run with --integration flag",
    )
    def test_search_integration(self) -> None:
        """Integration test for Chroma search."""
        pytest.skip("Requires running Chroma instance - run with --integration flag")
