"""Tests for Chroma semantic search pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.semantic_search.indexing.chroma import (
    ChromaSemanticIndexingPipeline,
)
from vectordb.haystack.semantic_search.search.chroma import (
    ChromaSemanticSearchPipeline,
)


class TestChromaSemanticSearch:
    """Unit tests for Chroma semantic search pipeline (indexing and search)."""

    # Indexing tests
    @patch("vectordb.haystack.semantic_search.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.chroma.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_init_loads_config(
        self,
        mock_get_docs: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = ChromaSemanticIndexingPipeline(chroma_config)
        assert pipeline.config == chroma_config
        assert pipeline.collection_name == "test_collection"

    @patch("vectordb.haystack.semantic_search.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.chroma.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_calls_create_collection(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method calls create_collection."""
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

        pipeline = ChromaSemanticIndexingPipeline(chroma_config)
        pipeline.run()

        mock_db.create_collection.assert_called_once()

    @patch("vectordb.haystack.semantic_search.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.chroma.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_returns_document_count(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run returns correct document count."""
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

        pipeline = ChromaSemanticIndexingPipeline(chroma_config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            ChromaSemanticIndexingPipeline(invalid_config)

    # Search tests
    @patch("vectordb.haystack.semantic_search.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.chroma.EmbedderFactory.create_text_embedder"
    )
    def test_search_init_loads_config(
        self,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = ChromaSemanticSearchPipeline(chroma_config)
        assert pipeline.config == chroma_config
        assert pipeline.collection_name == "test_collection"

    @patch("vectordb.haystack.semantic_search.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.chroma.EmbedderFactory.create_text_embedder"
    )
    def test_search_calls_embedder(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search method calls text embedder."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = ChromaSemanticSearchPipeline(chroma_config)
        pipeline.search("test query", top_k=5)

        mock_embedder.run.assert_called_once_with(text="test query")

    @patch("vectordb.haystack.semantic_search.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.chroma.EmbedderFactory.create_text_embedder"
    )
    def test_search_returns_documents(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search returns documents."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = ChromaSemanticSearchPipeline(chroma_config)
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert "query" in result

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            ChromaSemanticSearchPipeline(invalid_config)
