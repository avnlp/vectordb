"""Unit tests for Milvus JSON indexing pipelines."""

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.json_indexing import MilvusJSONIndexer, MilvusJSONSearcher


class TestMilvusJSONIndexing:
    """Test class for Milvus JSON indexing feature."""

    @pytest.fixture
    def config(self) -> dict[str, Any]:
        """Return test configuration for Milvus."""
        return {
            "milvus": {
                "uri": "http://localhost:19530",
                "collection_name": "test_collection",
            },
            "collection": {"name": "test_collection"},
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "dataloader": {"type": "triviaqa", "split": "test", "limit": 10},
            "search": {"top_k": 10},
        }

    @patch("vectordb.haystack.json_indexing.indexing.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.json_indexing.indexing.milvus.create_document_embedder")
    @patch("vectordb.haystack.json_indexing.indexing.milvus.get_embedding_dimension")
    @patch("vectordb.haystack.json_indexing.indexing.milvus.DataloaderCatalog")
    def test_indexing_unit(
        self,
        mock_registry: MagicMock,
        mock_get_dim: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        config: dict[str, Any],
    ) -> None:
        """Test indexing pipeline with mocked dependencies."""
        mock_documents = [
            Document(content="Test content 1", meta={"source": "test"}),
            Document(content="Test content 2", meta={"source": "test"}),
        ]
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = mock_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_registry.create.return_value = mock_loader

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {
            "documents": [
                Document(content="Test content 1", embedding=[0.1] * 384),
                Document(content="Test content 2", embedding=[0.2] * 384),
            ]
        }
        mock_embedder_factory.return_value = mock_embedder
        mock_get_dim.return_value = 384

        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        indexer = MilvusJSONIndexer(config)
        result = indexer.run()

        assert result["documents_indexed"] == 2
        mock_db.create_collection.assert_called_once()
        mock_db.insert_documents.assert_called_once()
        mock_registry.create.assert_called_once()

    @patch("vectordb.haystack.json_indexing.search.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.json_indexing.search.milvus.create_text_embedder")
    def test_search_unit(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        config: dict[str, Any],
    ) -> None:
        """Test search pipeline with mocked dependencies."""
        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = [
            {"content": "Result 1", "score": 0.9},
            {"content": "Result 2", "score": 0.8},
        ]
        mock_db_cls.return_value = mock_db

        searcher = MilvusJSONSearcher(config)
        results = searcher.search("test query")

        assert len(results) == 2
        assert results[0]["score"] == 0.9
        mock_db.search.assert_called_once()

    @patch("vectordb.haystack.json_indexing.search.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.json_indexing.search.milvus.create_text_embedder")
    def test_search_with_filters(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        config: dict[str, Any],
    ) -> None:
        """Test search with filter conditions."""
        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = [{"content": "Filtered result", "score": 0.95}]
        mock_db_cls.return_value = mock_db

        searcher = MilvusJSONSearcher(config)
        results = searcher.search("test query", filters={"category": "science"})

        assert len(results) == 1
        call_kwargs = mock_db.search.call_args[1]
        assert 'metadata["category"]' in call_kwargs["filter_expr"]
        assert '"science"' in call_kwargs["filter_expr"]

    @patch("vectordb.haystack.json_indexing.search.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.json_indexing.search.milvus.create_text_embedder")
    def test_search_with_custom_top_k(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        config: dict[str, Any],
    ) -> None:
        """Test search with custom top_k parameter."""
        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = []
        mock_db_cls.return_value = mock_db

        searcher = MilvusJSONSearcher(config)
        searcher.search("test query", top_k=5)

        call_kwargs = mock_db.search.call_args[1]
        assert call_kwargs["limit"] == 5

    @patch("vectordb.haystack.json_indexing.search.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.json_indexing.search.milvus.create_text_embedder")
    def test_search_uses_collection_from_config(
        self,
        mock_embedder_factory: MagicMock,
        mock_db_cls: MagicMock,
        config: dict[str, Any],
    ) -> None:
        """Test search uses collection name from config."""
        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_embedder_factory.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = []
        mock_db_cls.return_value = mock_db

        searcher = MilvusJSONSearcher(config)
        searcher.search("test query")

        call_kwargs = mock_db.search.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("MILVUS_URI"),
        reason="Requires running Milvus instance - run with --integration flag",
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Milvus indexing pipeline."""
        pytest.skip("Requires running Milvus instance - run with --integration flag")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("MILVUS_URI"),
        reason="Requires running Milvus instance - run with --integration flag",
    )
    def test_search_integration(self) -> None:
        """Integration test for Milvus search pipeline."""
        pytest.skip("Requires running Milvus instance - run with --integration flag")
