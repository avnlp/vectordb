"""Tests for Chroma metadata filtering pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.metadata_filtering.indexing.chroma import (
    ChromaMetadataFilteringIndexingPipeline,
)
from vectordb.haystack.metadata_filtering.search.chroma import (
    ChromaMetadataFilteringSearchPipeline,
)


class TestChromaMetadataFilteringIndexing:
    """Unit tests for Chroma metadata filtering indexing pipeline."""

    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.chroma.load_documents_from_config"
    )
    def test_indexing_init_loads_config(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = ChromaMetadataFilteringIndexingPipeline(chroma_config)
        assert pipeline.config == chroma_config
        assert pipeline.config["chroma"]["collection_name"] == "test_metadata"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.chroma.load_documents_from_config"
    )
    def test_indexing_run_calls_add_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test indexing run method calls add_documents."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.add_documents.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringIndexingPipeline(chroma_config)
        result = pipeline.run()

        mock_db.create_collection.assert_called_once()
        mock_db.add_documents.assert_called_once()
        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.chroma.load_documents_from_config"
    )
    def test_indexing_run_raises_on_empty_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test indexing run raises ValueError when no documents are loaded."""
        mock_load_docs.return_value = []

        pipeline = ChromaMetadataFilteringIndexingPipeline(chroma_config)

        with pytest.raises(ValueError, match="No documents loaded"):
            pipeline.run()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            ChromaMetadataFilteringIndexingPipeline(invalid_config)

    def test_indexing_missing_chroma_section(self, base_config: dict) -> None:
        """Test indexing fails when chroma section is missing."""
        with pytest.raises(ValueError, match="chroma"):
            ChromaMetadataFilteringIndexingPipeline(base_config)

    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.chroma.load_documents_from_config"
    )
    def test_indexing_creates_collection_with_correct_params(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test that collection is created with correct parameters."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.add_documents.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringIndexingPipeline(chroma_config)
        pipeline.run()

        mock_db.create_collection.assert_called_once_with(
            collection_name="test_metadata",
            recreate=False,
        )

    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.ChromaVectorDB")
    def test_init_db_uses_config_values(
        self,
        mock_db_class: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test _init_db uses correct config values."""
        ChromaMetadataFilteringIndexingPipeline(chroma_config)

        mock_db_class.assert_called_once_with(
            persist_directory="./test_chroma_data",
            collection_name="test_metadata",
        )

    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.chroma.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.chroma.load_documents_from_config"
    )
    def test_indexing_embeds_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test that documents are embedded before indexing."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.add_documents.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringIndexingPipeline(chroma_config)
        pipeline.run()

        mock_embedder.run.assert_called_once_with(documents=sample_documents)


class TestChromaMetadataFilteringSearch:
    """Unit tests for Chroma metadata filtering search pipeline."""

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.chroma.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.parse_filter_from_config"
    )
    @patch("vectordb.haystack.metadata_filtering.search.chroma.create_rag_generator")
    def test_search_init_loads_config(
        self,
        mock_rag: MagicMock,
        mock_filter: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = ChromaMetadataFilteringSearchPipeline(chroma_config)
        assert pipeline.config == chroma_config
        assert pipeline.config["chroma"]["collection_name"] == "test_metadata"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.chroma.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.chroma.create_rag_generator")
    def test_search_calls_query_method(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
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

        pipeline = ChromaMetadataFilteringSearchPipeline(chroma_config)
        results = pipeline.search("test query")

        mock_db.query.assert_called_once()
        assert len(results) == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.chroma.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.chroma.create_rag_generator")
    def test_search_returns_filtered_results(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search returns FilteredQueryResult objects with correct structure."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {"category": {"$eq": "ml"}}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        doc = Document(content="Test", meta={"category": "ml"})
        doc.score = 0.87
        mock_db = MagicMock()
        mock_db.query.return_value = [doc]
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringSearchPipeline(chroma_config)
        results = pipeline.search("test query")

        assert len(results) == 1
        assert results[0].relevance_score == 0.87
        assert results[0].rank == 1
        assert results[0].filter_matched is True
        assert results[0].timing is not None

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.chroma.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.chroma.create_rag_generator")
    def test_search_passes_where_filter_to_db(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search passes filter dict as 'where' param to database query method."""
        mock_rag.return_value = None
        filter_value = {"category": {"$eq": "ml"}}
        mock_filter_dict.return_value = filter_value

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringSearchPipeline(chroma_config)
        pipeline.search("test query")

        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["where"] == filter_value

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.chroma.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.chroma.create_rag_generator")
    def test_search_with_empty_filter(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search passes None when filter dict is empty."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringSearchPipeline(chroma_config)
        pipeline.search("test query")

        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["where"] is None

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.chroma.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.chroma.create_rag_generator")
    def test_search_uses_top_k_from_config(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search uses top_k value from config."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringSearchPipeline(chroma_config)
        pipeline.search("test query")

        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["top_k"] == 5

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            ChromaMetadataFilteringSearchPipeline(invalid_config)

    def test_search_missing_chroma_section(self, base_config: dict) -> None:
        """Test search fails when chroma section is missing."""
        with pytest.raises(ValueError, match="chroma"):
            ChromaMetadataFilteringSearchPipeline(base_config)

    def test_search_missing_search_section(self, chroma_config: dict) -> None:
        """Test search fails when search section is missing."""
        del chroma_config["search"]
        with pytest.raises(ValueError, match="search"):
            ChromaMetadataFilteringSearchPipeline(chroma_config)

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    def test_init_db_uses_config_values(
        self,
        mock_db_class: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test _init_db uses correct config values."""
        ChromaMetadataFilteringSearchPipeline(chroma_config)

        mock_db_class.assert_called_once_with(
            persist_directory="./test_chroma_data",
            collection_name="test_metadata",
        )

    @patch("vectordb.haystack.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.chroma.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.chroma.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.chroma.create_rag_generator")
    def test_search_handles_none_score(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        chroma_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search handles documents with None score."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        doc = Document(content="Test", meta={"category": "ml"})
        doc.score = None
        mock_db = MagicMock()
        mock_db.query.return_value = [doc]
        mock_db_class.return_value = mock_db

        pipeline = ChromaMetadataFilteringSearchPipeline(chroma_config)
        results = pipeline.search("test query")

        assert len(results) == 1
        assert results[0].relevance_score == 0.0
