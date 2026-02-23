"""Tests for indexing and search pipelines."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.metadata_filtering.indexing.milvus import (
    MilvusMetadataFilteringIndexingPipeline,
)
from vectordb.haystack.metadata_filtering.indexing.pinecone import (
    PineconeMetadataFilteringIndexingPipeline,
)
from vectordb.haystack.metadata_filtering.search.milvus import (
    MilvusMetadataFilteringSearchPipeline,
)
from vectordb.haystack.metadata_filtering.search.pinecone import (
    PineconeMetadataFilteringSearchPipeline,
)


class TestMilvusIndexingPipeline:
    """Test Milvus indexing pipeline."""

    @patch("vectordb.haystack.metadata_filtering.indexing.milvus.MilvusVectorDB")
    def test_init_with_dict_config(self, mock_db_class: MagicMock) -> None:
        """Test pipeline initialization with dict config."""
        config = {
            "dataloader": {"type": "triviaqa", "limit": 10},
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "milvus": {
                "uri": "http://localhost:19530",
                "collection_name": "test",
            },
        }

        mock_db_class.return_value = MagicMock()
        pipeline = MilvusMetadataFilteringIndexingPipeline(config)
        assert pipeline.config == config

    def test_init_missing_sections(self) -> None:
        """Test error when required config sections missing."""
        config = {"dataloader": {"type": "triviaqa"}}

        with pytest.raises(ValueError, match="embeddings"):
            MilvusMetadataFilteringIndexingPipeline(config)

    @patch("vectordb.haystack.metadata_filtering.indexing.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.metadata_filtering.indexing.milvus.get_document_embedder")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.milvus.load_documents_from_config"
    )
    def test_indexing_run(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
    ) -> None:
        """Test indexing pipeline run."""
        from haystack import Document

        config = {
            "dataloader": {"type": "triviaqa", "limit": 10},
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "milvus": {
                "uri": "http://localhost:19530",
                "collection_name": "test",
            },
        }

        # Mock document loading
        mock_load_docs.return_value = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
        ]

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {
            "documents": [
                Document(content="Doc 1", embedding=[0.1] * 384),
                Document(content="Doc 2", embedding=[0.2] * 384),
            ]
        }
        mock_get_embedder.return_value = mock_embedder

        # Mock VectorDB
        mock_db = MagicMock()
        mock_db.insert_documents.return_value = 2
        mock_db_class.return_value = mock_db

        pipeline = MilvusMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        mock_load_docs.assert_called_once()
        mock_get_embedder.assert_called_once()
        mock_db.create_collection.assert_called_once()
        mock_db.insert_documents.assert_called_once()


class TestPineconeIndexingPipeline:
    """Test Pinecone indexing pipeline."""

    @patch("vectordb.haystack.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    def test_init_with_dict_config(self, mock_db_class: MagicMock) -> None:
        """Test pipeline initialization with dict config."""
        config = {
            "dataloader": {"type": "triviaqa", "limit": 10},
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "pinecone": {"index_name": "test-index"},
        }

        mock_db_class.return_value = MagicMock()
        pipeline = PineconeMetadataFilteringIndexingPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.haystack.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.get_document_embedder"
    )
    @patch(
        "vectordb.haystack.metadata_filtering.indexing.pinecone.load_documents_from_config"
    )
    def test_indexing_run(
        self,
        mock_load_docs: MagicMock,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
    ) -> None:
        """Test indexing pipeline run."""
        from haystack import Document

        config = {
            "dataloader": {"type": "triviaqa", "limit": 10},
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "pinecone": {"index_name": "test-index"},
        }

        # Mock document loading
        mock_load_docs.return_value = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
        ]

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {
            "documents": [
                Document(content="Doc 1", embedding=[0.1] * 384),
                Document(content="Doc 2", embedding=[0.2] * 384),
            ]
        }
        mock_get_embedder.return_value = mock_embedder

        # Mock VectorDB
        mock_db = MagicMock()
        mock_db.upsert.return_value = 2
        mock_db_class.return_value = mock_db

        pipeline = PineconeMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 2
        mock_load_docs.assert_called_once()
        mock_get_embedder.assert_called_once()
        mock_db.create_index.assert_called_once()
        mock_db.upsert.assert_called_once()


class TestMilvusSearchPipeline:
    """Test Milvus search pipeline."""

    @patch("vectordb.haystack.metadata_filtering.search.milvus.MilvusVectorDB")
    def test_init_with_dict_config(self, mock_db_class: MagicMock) -> None:
        """Test pipeline initialization with dict config."""
        config = {
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "milvus": {
                "uri": "http://localhost:19530",
                "collection_name": "test",
            },
            "search": {"top_k": 10},
        }

        mock_db_class.return_value = MagicMock()
        pipeline = MilvusMetadataFilteringSearchPipeline(config)
        assert pipeline.config == config

    def test_init_missing_sections(self) -> None:
        """Test error when required config sections missing."""
        config = {
            "embeddings": {"model": "test-model"},
            "search": {"top_k": 10},
        }

        with pytest.raises(ValueError, match="milvus"):
            MilvusMetadataFilteringSearchPipeline(config)

    @patch("vectordb.haystack.metadata_filtering.search.milvus.MilvusVectorDB")
    @patch("vectordb.haystack.metadata_filtering.search.milvus.get_text_embedder")
    def test_search_run(
        self,
        mock_get_embedder: MagicMock,
        mock_db_class: MagicMock,
    ) -> None:
        """Test search pipeline run."""
        from haystack import Document

        config = {
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "milvus": {
                "uri": "http://localhost:19530",
                "collection_name": "test",
            },
            "search": {"top_k": 10},
            "metadata_filtering": {"test_query": "test query"},
        }

        # Mock text embedder
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_get_embedder.return_value = mock_embedder

        # Mock VectorDB search
        mock_db = MagicMock()
        doc1 = Document(content="Result 1")
        doc1.score = 0.95
        mock_db.search.return_value = [doc1]
        mock_db_class.return_value = mock_db

        pipeline = MilvusMetadataFilteringSearchPipeline(config)
        results = pipeline.search()

        assert len(results) == 1
        assert results[0].document.content == "Result 1"
        assert results[0].relevance_score == 0.95
        assert results[0].rank == 1


class TestPineconeSearchPipeline:
    """Test Pinecone search pipeline."""

    @patch("vectordb.haystack.metadata_filtering.search.pinecone.PineconeVectorDB")
    def test_init_with_dict_config(self, mock_db_class: MagicMock) -> None:
        """Test pipeline initialization with dict config."""
        config = {
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "pinecone": {"index_name": "test-index"},
            "search": {"top_k": 10},
        }

        mock_db_class.return_value = MagicMock()
        pipeline = PineconeMetadataFilteringSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.haystack.metadata_filtering.search.pinecone.get_text_embedder")
    @patch("vectordb.haystack.metadata_filtering.search.pinecone.PineconeVectorDB")
    def test_search_run(
        self,
        mock_db_class: MagicMock,
        mock_get_embedder: MagicMock,
    ) -> None:
        """Test search pipeline run."""
        from haystack import Document

        config = {
            "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "pinecone": {"index_name": "test-index"},
            "search": {"top_k": 10},
            "metadata_filtering": {"test_query": "test query"},
        }

        # Mock text embedder
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_get_embedder.return_value = mock_embedder

        # Mock VectorDB query
        mock_db = MagicMock()
        doc1 = Document(content="Result 1")
        doc1.score = 0.95
        mock_db.query.return_value = [doc1]
        mock_db_class.return_value = mock_db

        pipeline = PineconeMetadataFilteringSearchPipeline(config)
        results = pipeline.search()

        assert len(results) == 1
        assert results[0].document.content == "Result 1"
        assert results[0].relevance_score == 0.95
        assert results[0].rank == 1
