"""Tests for Pinecone reranking pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.reranking.indexing.pinecone import (
    PineconeRerankingIndexingPipeline,
)
from vectordb.haystack.reranking.search.pinecone import PineconeRerankingSearchPipeline


class TestPineconeReranking:
    """Unit tests for Pinecone reranking pipeline (indexing and search)."""

    # Indexing tests
    @patch(
        "vectordb.haystack.reranking.indexing.pinecone.EmbedderFactory.get_embedding_dimension"
    )
    @patch("vectordb.haystack.reranking.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.reranking.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_init_loads_config(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        mock_dimension: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        mock_dimension.return_value = 384
        pipeline = PineconeRerankingIndexingPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.dimension == 384

    @patch(
        "vectordb.haystack.reranking.indexing.pinecone.EmbedderFactory.get_embedding_dimension"
    )
    @patch("vectordb.haystack.reranking.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.reranking.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_run_calls_create_index(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        mock_dimension: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method calls create_index."""
        mock_dimension.return_value = 384
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

        pipeline = PineconeRerankingIndexingPipeline(pinecone_config)
        pipeline.run()

        mock_db.create_index.assert_called_once()

    @patch(
        "vectordb.haystack.reranking.indexing.pinecone.EmbedderFactory.get_embedding_dimension"
    )
    @patch("vectordb.haystack.reranking.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.indexing.pinecone.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.reranking.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_run_calls_upsert(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        mock_dimension: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method calls upsert."""
        mock_dimension.return_value = 384
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

        pipeline = PineconeRerankingIndexingPipeline(pinecone_config)
        pipeline.run()

        mock_db.upsert.assert_called_once()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            PineconeRerankingIndexingPipeline(invalid_config)

    # Search tests
    @patch("vectordb.haystack.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.pinecone.RerankerFactory.create")
    def test_search_init_loads_config(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = PineconeRerankingSearchPipeline(pinecone_config)
        assert pipeline.config == pinecone_config

    @patch("vectordb.haystack.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.pinecone.RerankerFactory.create")
    def test_search_calls_query(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search method calls query."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_reranker = MagicMock()
        mock_reranker.run.return_value = {"documents": sample_documents}
        mock_make_reranker.return_value = mock_reranker

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = PineconeRerankingSearchPipeline(pinecone_config)
        pipeline.search("test query", top_k=5)

        mock_db.query.assert_called_once()

    @patch("vectordb.haystack.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.pinecone.RerankerFactory.create")
    def test_search_returns_documents(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search returns documents."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_reranker = MagicMock()
        mock_reranker.run.return_value = {"documents": sample_documents}
        mock_make_reranker.return_value = mock_reranker

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = PineconeRerankingSearchPipeline(pinecone_config)
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert "query" in result

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            PineconeRerankingSearchPipeline(invalid_config)

    # Additional search method tests
    @patch("vectordb.haystack.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.pinecone.RerankerFactory.create")
    def test_search_handles_empty_results(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test search handles empty results from database."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedding = [0.1] * 384
        mock_embedder.run.return_value = {"embedding": mock_embedding}
        mock_make_embedder.return_value = mock_embedder

        # Setup mock reranker
        mock_reranker = MagicMock()
        mock_make_reranker.return_value = mock_reranker

        # Setup mock database to return empty list
        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_class.return_value = mock_db

        # Create pipeline and search
        pipeline = PineconeRerankingSearchPipeline(pinecone_config)
        result = pipeline.search(query="test query")

        # Verify empty results are handled
        assert result["query"] == "test query"
        assert result["documents"] == []
        mock_reranker.run.assert_not_called()

    @patch("vectordb.haystack.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.pinecone.RerankerFactory.create")
    def test_search_with_custom_top_k(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test search with custom top_k parameter."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedding = [0.1] * 384
        mock_embedder.run.return_value = {"embedding": mock_embedding}
        mock_make_embedder.return_value = mock_embedder

        # Setup mock reranker
        mock_reranker = MagicMock()
        mock_reranker.run.return_value = {"documents": sample_documents[:3]}
        mock_make_reranker.return_value = mock_reranker

        # Setup mock database
        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        # Create pipeline and search with custom top_k
        pipeline = PineconeRerankingSearchPipeline(pinecone_config)
        result = pipeline.search(query="test query", top_k=3)

        # Verify top_k was passed correctly
        assert result["documents"] == sample_documents[:3]
        # Verify retrieval uses 3x top_k for reranking
        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        assert call_args.kwargs["top_k"] == 9  # 3 * 3

    @patch("vectordb.haystack.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.pinecone.RerankerFactory.create")
    def test_search_with_filters(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test search with metadata filters."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedding = [0.1] * 384
        mock_embedder.run.return_value = {"embedding": mock_embedding}
        mock_make_embedder.return_value = mock_embedder

        # Setup mock reranker
        mock_reranker = MagicMock()
        mock_reranker.run.return_value = {"documents": sample_documents}
        mock_make_reranker.return_value = mock_reranker

        # Setup mock database
        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        # Create pipeline and search with filters
        pipeline = PineconeRerankingSearchPipeline(pinecone_config)
        filters = {"source": "wiki"}
        pipeline.search(query="test query", filters=filters)

        # Verify filters were passed to database
        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        assert call_args.kwargs.get("filter") == filters

    @patch("vectordb.haystack.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.pinecone.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.pinecone.RerankerFactory.create")
    def test_run_alias_for_search(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test run method is alias for search."""
        # Setup mock embedder
        mock_embedder = MagicMock()
        mock_embedding = [0.1] * 384
        mock_embedder.run.return_value = {"embedding": mock_embedding}
        mock_make_embedder.return_value = mock_embedder

        # Setup mock reranker
        mock_reranker = MagicMock()
        mock_reranker.run.return_value = {"documents": sample_documents}
        mock_make_reranker.return_value = mock_reranker

        # Setup mock database
        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        # Create pipeline and use run method
        pipeline = PineconeRerankingSearchPipeline(pinecone_config)
        result = pipeline.run(query="test query", top_k=5)

        # Verify run returns same results as search
        assert result == {"query": "test query", "documents": sample_documents}
