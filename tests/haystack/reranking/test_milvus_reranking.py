"""Tests for Milvus reranking pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.reranking.indexing.milvus import MilvusRerankingIndexingPipeline
from vectordb.haystack.reranking.search.milvus import MilvusRerankingSearchPipeline


class TestMilvusReranking:
    """Unit tests for Milvus reranking pipeline (indexing and search)."""

    # Indexing tests
    @patch(
        "vectordb.haystack.reranking.indexing.milvus.EmbedderFactory.get_embedding_dimension"
    )
    @patch("vectordb.haystack.reranking.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.indexing.milvus.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.reranking.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_init_loads_config(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        mock_dimension: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        mock_dimension.return_value = 384
        pipeline = MilvusRerankingIndexingPipeline(milvus_config)
        assert pipeline.config == milvus_config
        assert pipeline.dimension == 384

    @patch(
        "vectordb.haystack.reranking.indexing.milvus.EmbedderFactory.get_embedding_dimension"
    )
    @patch("vectordb.haystack.reranking.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.indexing.milvus.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.reranking.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_run_handles_documents(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        mock_dimension: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method processes documents."""
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

        pipeline = MilvusRerankingIndexingPipeline(milvus_config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db.upsert.assert_called_once()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            MilvusRerankingIndexingPipeline(invalid_config)

    # Search tests
    @patch("vectordb.haystack.reranking.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.milvus.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.milvus.RerankerFactory.create")
    def test_search_init_loads_config(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = MilvusRerankingSearchPipeline(milvus_config)
        assert pipeline.config == milvus_config

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            MilvusRerankingSearchPipeline(invalid_config)

    # Search method tests
    @patch("vectordb.haystack.reranking.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.milvus.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.milvus.RerankerFactory.create")
    def test_search_returns_reranked_documents(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test search method returns reranked documents."""
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
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        # Create pipeline and search
        pipeline = MilvusRerankingSearchPipeline(milvus_config)
        result = pipeline.search(query="test query", top_k=5)

        # Verify results
        assert result["query"] == "test query"
        assert result["documents"] == sample_documents
        mock_embedder.run.assert_called_once_with(text="test query")
        mock_db.search.assert_called_once()
        mock_reranker.run.assert_called_once()

    @patch("vectordb.haystack.reranking.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.milvus.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.milvus.RerankerFactory.create")
    def test_search_handles_empty_results(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
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
        mock_db.search.return_value = []
        mock_db_class.return_value = mock_db

        # Create pipeline and search
        pipeline = MilvusRerankingSearchPipeline(milvus_config)
        result = pipeline.search(query="test query")

        # Verify empty results are handled
        assert result["query"] == "test query"
        assert result["documents"] == []
        mock_reranker.run.assert_not_called()

    @patch("vectordb.haystack.reranking.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.milvus.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.milvus.RerankerFactory.create")
    def test_search_with_custom_top_k(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
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
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        # Create pipeline and search with custom top_k
        pipeline = MilvusRerankingSearchPipeline(milvus_config)
        result = pipeline.search(query="test query", top_k=3)

        # Verify top_k was passed correctly
        assert result["documents"] == sample_documents[:3]
        # Verify retrieval uses 3x top_k for reranking
        mock_db.search.assert_called_once()
        call_args = mock_db.search.call_args
        assert call_args.kwargs["top_k"] == 9  # 3 * 3

    @patch("vectordb.haystack.reranking.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.milvus.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.milvus.RerankerFactory.create")
    def test_search_with_filters(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
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
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        # Create pipeline and search with filters
        pipeline = MilvusRerankingSearchPipeline(milvus_config)
        filters = {"source": "wiki"}
        pipeline.search(query="test query", filters=filters)

        # Verify filters were passed to database
        mock_db.search.assert_called_once()
        call_args = mock_db.search.call_args
        assert call_args.kwargs["filters"] == filters

    @patch("vectordb.haystack.reranking.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.reranking.search.milvus.EmbedderFactory.create_text_embedder"
    )
    @patch("vectordb.haystack.reranking.search.milvus.RerankerFactory.create")
    def test_run_alias_for_search(
        self,
        mock_make_reranker: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
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
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        # Create pipeline and use run method
        pipeline = MilvusRerankingSearchPipeline(milvus_config)
        result = pipeline.run(query="test query", top_k=5)

        # Verify run returns same results as search
        assert result == {"query": "test query", "documents": sample_documents}
