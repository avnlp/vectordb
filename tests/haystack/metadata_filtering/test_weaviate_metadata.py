"""Tests for Weaviate metadata filtering pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.metadata_filtering.indexing.weaviate import (
    WeaviateMetadataFilteringIndexingPipeline,
)
from vectordb.haystack.metadata_filtering.search.weaviate import (
    WeaviateMetadataFilteringSearchPipeline,
)


class TestWeaviateMetadataFilteringIndexing:
    """Unit tests for Weaviate metadata filtering indexing pipeline."""

    @patch("vectordb.haystack.metadata_filtering.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.load_documents_from_config"
    )
    def test_indexing_init_loads_config(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = WeaviateMetadataFilteringIndexingPipeline(weaviate_config)
        assert pipeline.config == weaviate_config
        assert pipeline.config["weaviate"]["collection_name"] == "TestMetadata"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.load_documents_from_config"
    )
    def test_indexing_run_calls_insert_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test indexing run method calls insert_documents."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.insert_documents.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = WeaviateMetadataFilteringIndexingPipeline(weaviate_config)
        result = pipeline.run()

        mock_db.create_collection.assert_called_once()
        mock_db.insert_documents.assert_called_once()
        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.load_documents_from_config"
    )
    def test_indexing_run_raises_on_empty_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test indexing run raises ValueError when no documents are loaded."""
        mock_load_docs.return_value = []

        pipeline = WeaviateMetadataFilteringIndexingPipeline(weaviate_config)

        with pytest.raises(ValueError, match="No documents loaded"):
            pipeline.run()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            WeaviateMetadataFilteringIndexingPipeline(invalid_config)

    def test_indexing_missing_weaviate_section(self, base_config: dict) -> None:
        """Test indexing fails when weaviate section is missing."""
        with pytest.raises(ValueError, match="weaviate"):
            WeaviateMetadataFilteringIndexingPipeline(base_config)

    @patch("vectordb.haystack.metadata_filtering.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.weaviate.load_documents_from_config"
    )
    def test_indexing_creates_collection_with_correct_params(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test that collection is created with correct parameters."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.insert_documents.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = WeaviateMetadataFilteringIndexingPipeline(weaviate_config)
        pipeline.run()

        mock_db.create_collection.assert_called_once_with(
            collection_name="TestMetadata",
            recreate=False,
        )

    @patch("vectordb.haystack.metadata_filtering.indexing.weaviate.WeaviateVectorDB")
    def test_init_db_uses_config_values(
        self,
        mock_db_class: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test _init_db uses correct config values."""
        WeaviateMetadataFilteringIndexingPipeline(weaviate_config)

        mock_db_class.assert_called_once_with(
            url="http://localhost:8080",
            api_key="",
            collection_name="TestMetadata",
        )


class TestWeaviateMetadataFilteringSearch:
    """Unit tests for Weaviate metadata filtering search pipeline."""

    @patch("vectordb.haystack.metadata_filtering.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.parse_filter_from_config"
    )
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.create_rag_generator")
    def test_search_init_loads_config(
        self,
        mock_rag: MagicMock,
        mock_filter: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = WeaviateMetadataFilteringSearchPipeline(weaviate_config)
        assert pipeline.config == weaviate_config
        assert pipeline.config["weaviate"]["collection_name"] == "TestMetadata"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.create_rag_generator")
    def test_search_calls_search_method(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        sample_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search method calls search on the database."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {"category": {"$eq": "ml"}}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = WeaviateMetadataFilteringSearchPipeline(weaviate_config)
        results = pipeline.search("test query")

        mock_db.search.assert_called_once()
        assert len(results) == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.create_rag_generator")
    def test_search_returns_filtered_results(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search returns FilteredQueryResult objects with correct structure."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {"category": {"$eq": "ml"}}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        doc = Document(content="Test", meta={"category": "ml"})
        doc.score = 0.91
        mock_db = MagicMock()
        mock_db.search.return_value = [doc]
        mock_db_class.return_value = mock_db

        pipeline = WeaviateMetadataFilteringSearchPipeline(weaviate_config)
        results = pipeline.search("test query")

        assert len(results) == 1
        assert results[0].relevance_score == 0.91
        assert results[0].rank == 1
        assert results[0].filter_matched is True
        assert results[0].timing is not None

    @patch("vectordb.haystack.metadata_filtering.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.create_rag_generator")
    def test_search_passes_filter_to_db(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search passes filter dict to database search method."""
        mock_rag.return_value = None
        filter_value = {"category": {"$eq": "ml"}}
        mock_filter_dict.return_value = filter_value

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = WeaviateMetadataFilteringSearchPipeline(weaviate_config)
        pipeline.search("test query")

        call_kwargs = mock_db.search.call_args[1]
        assert call_kwargs["filters"] == filter_value

    @patch("vectordb.haystack.metadata_filtering.search.weaviate.WeaviateVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.weaviate.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.weaviate.create_rag_generator")
    def test_search_uses_top_k_from_config(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        weaviate_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search uses top_k value from config."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = WeaviateMetadataFilteringSearchPipeline(weaviate_config)
        pipeline.search("test query")

        call_kwargs = mock_db.search.call_args[1]
        assert call_kwargs["top_k"] == 5

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            WeaviateMetadataFilteringSearchPipeline(invalid_config)

    def test_search_missing_weaviate_section(self, base_config: dict) -> None:
        """Test search fails when weaviate section is missing."""
        with pytest.raises(ValueError, match="weaviate"):
            WeaviateMetadataFilteringSearchPipeline(base_config)

    def test_search_missing_search_section(self, weaviate_config: dict) -> None:
        """Test search fails when search section is missing."""
        del weaviate_config["search"]
        with pytest.raises(ValueError, match="search"):
            WeaviateMetadataFilteringSearchPipeline(weaviate_config)

    @patch("vectordb.haystack.metadata_filtering.search.weaviate.WeaviateVectorDB")
    def test_init_db_uses_config_values(
        self,
        mock_db_class: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test _init_db uses correct config values."""
        WeaviateMetadataFilteringSearchPipeline(weaviate_config)

        mock_db_class.assert_called_once_with(
            url="http://localhost:8080",
            api_key="",
            collection_name="TestMetadata",
        )
