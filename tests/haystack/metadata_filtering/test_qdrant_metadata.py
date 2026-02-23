"""Tests for Qdrant metadata filtering pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.metadata_filtering.indexing.qdrant import (
    QdrantMetadataFilteringIndexingPipeline,
)
from vectordb.haystack.metadata_filtering.search.qdrant import (
    QdrantMetadataFilteringSearchPipeline,
)


class TestQdrantMetadataFilteringIndexing:
    """Unit tests for Qdrant metadata filtering indexing pipeline."""

    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.qdrant.load_documents_from_config"
    )
    def test_indexing_init_loads_config(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = QdrantMetadataFilteringIndexingPipeline(qdrant_config)
        assert pipeline.config == qdrant_config
        assert pipeline.config["qdrant"]["collection_name"] == "test_metadata"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.qdrant.load_documents_from_config"
    )
    def test_indexing_run_calls_index_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test indexing run method calls index_documents."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.index_documents.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = QdrantMetadataFilteringIndexingPipeline(qdrant_config)
        result = pipeline.run()

        mock_db.create_collection.assert_called_once()
        mock_db.index_documents.assert_called_once()
        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.qdrant.load_documents_from_config"
    )
    def test_indexing_run_raises_on_empty_documents(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test indexing run raises ValueError when no documents are loaded."""
        mock_load_docs.return_value = []

        pipeline = QdrantMetadataFilteringIndexingPipeline(qdrant_config)

        with pytest.raises(ValueError, match="No documents loaded"):
            pipeline.run()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            QdrantMetadataFilteringIndexingPipeline(invalid_config)

    def test_indexing_missing_qdrant_section(self, base_config: dict) -> None:
        """Test indexing fails when qdrant section is missing."""
        with pytest.raises(ValueError, match="qdrant"):
            QdrantMetadataFilteringIndexingPipeline(base_config)

    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.qdrant.load_documents_from_config"
    )
    def test_indexing_creates_collection_with_correct_params(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
        sample_documents: list[Document],
    ) -> None:
        """Test that collection is created with correct parameters."""
        mock_load_docs.return_value = sample_documents

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_get_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.index_documents.return_value = len(sample_documents)
        mock_db_class.return_value = mock_db

        pipeline = QdrantMetadataFilteringIndexingPipeline(qdrant_config)
        pipeline.run()

        mock_db.create_collection.assert_called_once_with(
            collection_name="test_metadata",
            dimension=384,
            recreate=False,
        )

    @patch("vectordb.haystack.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    def test_init_db_uses_config_values(
        self,
        mock_db_class: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test _init_db uses correct config values."""
        QdrantMetadataFilteringIndexingPipeline(qdrant_config)

        mock_db_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key="",
            collection_name="test_metadata",
        )


class TestQdrantMetadataFilteringSearch:
    """Unit tests for Qdrant metadata filtering search pipeline."""

    @patch("vectordb.haystack.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.parse_filter_from_config"
    )
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.create_rag_generator")
    def test_search_init_loads_config(
        self,
        mock_rag: MagicMock,
        mock_filter: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        qdrant_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = QdrantMetadataFilteringSearchPipeline(qdrant_config)
        assert pipeline.config == qdrant_config
        assert pipeline.config["qdrant"]["collection_name"] == "test_metadata"
        mock_db.assert_called_once()

    @patch("vectordb.haystack.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.create_rag_generator")
    def test_search_calls_search_method(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
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

        pipeline = QdrantMetadataFilteringSearchPipeline(qdrant_config)
        results = pipeline.search("test query")

        mock_db.search.assert_called_once()
        assert len(results) == len(sample_documents)

    @patch("vectordb.haystack.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.create_rag_generator")
    def test_search_returns_filtered_results(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search returns FilteredQueryResult objects with correct structure."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {"category": {"$eq": "ml"}}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        doc = Document(content="Test", meta={"category": "ml"})
        doc.score = 0.88
        mock_db = MagicMock()
        mock_db.search.return_value = [doc]
        mock_db_class.return_value = mock_db

        pipeline = QdrantMetadataFilteringSearchPipeline(qdrant_config)
        results = pipeline.search("test query")

        assert len(results) == 1
        assert results[0].relevance_score == 0.88
        assert results[0].rank == 1
        assert results[0].filter_matched is True
        assert results[0].timing is not None

    @patch("vectordb.haystack.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.create_rag_generator")
    def test_search_passes_filter_to_db(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
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

        pipeline = QdrantMetadataFilteringSearchPipeline(qdrant_config)
        pipeline.search("test query")

        call_kwargs = mock_db.search.call_args[1]
        assert call_kwargs["filters"] == filter_value

    @patch("vectordb.haystack.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.get_text_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.parse_filter_from_config"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.search.qdrant.filter_spec_to_canonical_dict"
    )
    @patch("vectordb.haystack.metadata_filtering.search.qdrant.create_rag_generator")
    def test_search_handles_multiple_results(
        self,
        mock_rag: MagicMock,
        mock_filter_dict: MagicMock,
        mock_filter: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
        qdrant_config: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search correctly ranks multiple results."""
        mock_rag.return_value = None
        mock_filter_dict.return_value = {}

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_get_embedder.return_value = mock_embedder

        docs = []
        for i in range(3):
            doc = Document(content=f"Test {i}", meta={"id": str(i)})
            doc.score = 0.9 - (i * 0.1)
            docs.append(doc)

        mock_db = MagicMock()
        mock_db.search.return_value = docs
        mock_db_class.return_value = mock_db

        pipeline = QdrantMetadataFilteringSearchPipeline(qdrant_config)
        results = pipeline.search("test query")

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.rank == i + 1
            assert result.timing is None or i == 0

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError, match="Missing or empty"):
            QdrantMetadataFilteringSearchPipeline(invalid_config)

    def test_search_missing_qdrant_section(self, base_config: dict) -> None:
        """Test search fails when qdrant section is missing."""
        with pytest.raises(ValueError, match="qdrant"):
            QdrantMetadataFilteringSearchPipeline(base_config)

    def test_search_missing_search_section(self, qdrant_config: dict) -> None:
        """Test search fails when search section is missing."""
        del qdrant_config["search"]
        with pytest.raises(ValueError, match="search"):
            QdrantMetadataFilteringSearchPipeline(qdrant_config)
