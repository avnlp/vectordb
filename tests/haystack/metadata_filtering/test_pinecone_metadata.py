"""Tests for Pinecone metadata filtering pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.metadata_filtering.indexing.pinecone import (
    PineconeMetadataFilteringIndexingPipeline,
)
from vectordb.haystack.metadata_filtering.search.pinecone import (
    PineconeMetadataFilteringSearchPipeline,
)


class TestPineconeMetadataFilteringIndexing:
    """Unit tests for Pinecone metadata filtering indexing pipeline."""

    @patch("vectordb.haystack.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.load_documents_from_config"
    )
    def test_indexing_init_loads_config(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = PineconeMetadataFilteringIndexingPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.config["pinecone"]["index_name"] == "test-metadata-index"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.load_documents_from_config"
    )
    def test_indexing_run_calls_upsert(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test indexing run method calls upsert."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.upsert.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = PineconeMetadataFilteringIndexingPipeline(pinecone_config)
        result = pipeline.run()

        mock_db.create_index.assert_called_once()
        mock_db.upsert.assert_called_once()
        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.load_documents_from_config"
    )
    def test_indexing_run_raises_on_empty_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test indexing run raises ValueError when no documents are loaded."""
        mock_load_docs.return_value = []

        pipeline = PineconeMetadataFilteringIndexingPipeline(pinecone_config)

        with pytest.raises(ValueError, match="No documents loaded"):
            pipeline.run()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            PineconeMetadataFilteringIndexingPipeline(invalid_config)

    def test_indexing_missing_pinecone_section(self, base_config: dict) -> None:
        """Test indexing fails when pinecone section is missing."""
        with pytest.raises(ValueError, match="pinecone"):
            PineconeMetadataFilteringIndexingPipeline(base_config)


class TestPineconeMetadataFilteringSearch:
    """Unit tests for Pinecone metadata filtering search pipeline."""

    @patch("vectordb.haystack.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.pinecone.parse_filter_from_config"
    )
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.create_rag_generator")
    def test_search_init_loads_config(
        self,
        mock_rag: MagicMock,
        mock_filter: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        pinecone_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = PineconeMetadataFilteringSearchPipeline(pinecone_config)
        assert pipeline.config == pinecone_config
        assert pipeline.config["pinecone"]["index_name"] == "test-metadata-index"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.pinecone.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.pinecone.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.create_rag_generator")
    def test_search_calls_query(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search method calls query on the database."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {"category": {"$eq": "ml"}}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = PineconeMetadataFilteringSearchPipeline(pinecone_config)
        results = pipeline.search("test query")

        mock_db.query.assert_called_once()
        assert len(results) == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.pinecone.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.pinecone.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.create_rag_generator")
    def test_search_returns_filtered_results(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search returns FilteredQueryResult objects with correct structure."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {"category": {"$eq": "ml"}}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        doc = Document(content="Test", meta={"category": "ml"})
        doc.score = 0.95
        mock_db = MagicMock()
        mock_db.query.return_value = [doc]
        mock_db_class.return_value = mock_db

        pipeline = PineconeMetadataFilteringSearchPipeline(pinecone_config)
        results = pipeline.search("test query")

        assert len(results) == 1
        assert results[0].relevance_score == 0.95
        assert results[0].rank == 1
        assert results[0].filter_matched is True
        assert results[0].timing is not None

    @patch("vectordb.haystack.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.pinecone.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.pinecone.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.create_rag_generator")
    def test_search_uses_config_query_when_none_provided(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        pinecone_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search uses test_query from config when no query is provided."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = PineconeMetadataFilteringSearchPipeline(pinecone_config)
        pipeline.search()

        mock_embedder.run.assert_called_once()
        call_args = mock_embedder.run.call_args
        assert call_args[1]["text"] == "What is machine learning?"

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            PineconeMetadataFilteringSearchPipeline(invalid_config)

    def test_search_missing_pinecone_section(self, base_config: dict) -> None:
        """Test search fails when pinecone section is missing."""
        with pytest.raises(ValueError, match="pinecone"):
            PineconeMetadataFilteringSearchPipeline(base_config)

    def test_search_missing_search_section(self, pinecone_config: dict) -> None:
        """Test search fails when search section is missing."""
        del pinecone_config["search"]
        with pytest.raises(ValueError, match="search"):
            PineconeMetadataFilteringSearchPipeline(pinecone_config)
