"""Tests for Pinecone semantic search pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.semantic_search.indexing.pinecone import (
    PineconeSemanticIndexingPipeline,
)
from vectordb.haystack.semantic_search.search.pinecone import (
    PineconeSemanticSearchPipeline,
)


class TestPineconeSemanticSearch:
    """Unit tests for Pinecone semantic search pipeline (indexing and search)."""

    # Indexing tests
    @patch("vectordb.haystack.semantic_search.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch(
        "vectordb.haystack.semantic_search.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_init_loads_config(
        self,
        mock_get_docs: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = PineconeSemanticIndexingPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.dimension == 384

    @patch("vectordb.haystack.semantic_search.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch(
        "vectordb.haystack.semantic_search.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_calls_create_index(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method calls create_index."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = PineconeSemanticIndexingPipeline(pinecone_config)
        pipeline.run()

        mock_db.create_index.assert_called_once()

    @patch("vectordb.haystack.semantic_search.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch(
        "vectordb.haystack.semantic_search.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_calls_upsert(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method calls upsert."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.upsert.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = PineconeSemanticIndexingPipeline(pinecone_config)
        pipeline.run()

        mock_db.upsert.assert_called_once()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            PineconeSemanticIndexingPipeline(invalid_config)

    # Search tests
    @patch("vectordb.haystack.semantic_search.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    def test_search_init_loads_config(
        self,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = PineconeSemanticSearchPipeline(pinecone_config)
        assert pipeline.config == pinecone_config

    @patch("vectordb.haystack.semantic_search.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    def test_search_calls_query(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search method calls query."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = PineconeSemanticSearchPipeline(pinecone_config)
        pipeline.search("test query", top_k=5)

        mock_db.query.assert_called_once()

    @patch("vectordb.haystack.semantic_search.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    def test_search_returns_documents(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search returns documents."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = PineconeSemanticSearchPipeline(pinecone_config)
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert "query" in result

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            PineconeSemanticSearchPipeline(invalid_config)
