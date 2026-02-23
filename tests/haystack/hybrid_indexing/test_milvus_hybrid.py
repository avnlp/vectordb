"""Unit tests for Milvus hybrid indexing and search pipelines."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document


class TestMilvusHybridIndexing:
    """Tests for MilvusHybridIndexingPipeline."""

    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.MilvusVectorDB")
    def test_init_loads_config(
        self,
        mock_milvus_db: MagicMock,
        mock_embedder_factory: MagicMock,
        milvus_config: dict[str, Any],
    ) -> None:
        """Test that __init__ loads and validates configuration."""
        from vectordb.haystack.hybrid_indexing.indexing.milvus import (
            MilvusHybridIndexingPipeline,
        )

        mock_embedder_factory.create_document_embedder.return_value = MagicMock()
        mock_embedder_factory.create_sparse_document_embedder.return_value = MagicMock()

        pipeline = MilvusHybridIndexingPipeline(milvus_config)

        assert pipeline.config == milvus_config
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.dimension == 384
        mock_milvus_db.assert_called_once_with(uri="http://localhost:19530", token="")
        mock_embedder_factory.create_document_embedder.assert_called_once()

    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.DataloaderCatalog.create")
    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.MilvusVectorDB")
    def test_run_calls_insert_documents(
        self,
        mock_milvus_db: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_get_docs: MagicMock,
        milvus_config: dict[str, Any],
        sample_documents: list[Document],
    ) -> None:
        """Test that run() loads documents, embeds, and inserts into Milvus."""
        from vectordb.haystack.hybrid_indexing.indexing.milvus import (
            MilvusHybridIndexingPipeline,
        )

        mock_db_instance = MagicMock()
        mock_milvus_db.return_value = mock_db_instance

        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {"documents": sample_documents}
        mock_embedder_factory.create_document_embedder.return_value = (
            mock_dense_embedder
        )
        mock_embedder_factory.create_sparse_document_embedder.return_value = None

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        pipeline = MilvusHybridIndexingPipeline(milvus_config)
        result = pipeline.run()

        mock_loader.load.assert_called_once()
        mock_dense_embedder.run.assert_called_once_with(documents=sample_documents)
        mock_db_instance.create_collection.assert_called_once()
        mock_db_instance.upsert.assert_called()
        assert result["documents_indexed"] == 3
        assert result["db"] == "milvus"
        assert result["collection_name"] == "test_hybrid"

    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.DataloaderCatalog.create")
    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.indexing.milvus.MilvusVectorDB")
    def test_run_with_sparse_embeddings(
        self,
        mock_milvus_db: MagicMock,
        mock_embedder_factory: MagicMock,
        mock_get_docs: MagicMock,
        milvus_config: dict[str, Any],
        sample_documents_with_sparse: list[Document],
    ) -> None:
        """Test that run() uses sparse embedder when configured."""
        from vectordb.haystack.hybrid_indexing.indexing.milvus import (
            MilvusHybridIndexingPipeline,
        )

        mock_db_instance = MagicMock()
        mock_milvus_db.return_value = mock_db_instance

        mock_dense_embedder = MagicMock()
        mock_dense_embedder.run.return_value = {
            "documents": sample_documents_with_sparse
        }
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {
            "documents": sample_documents_with_sparse
        }

        mock_embedder_factory.create_document_embedder.return_value = (
            mock_dense_embedder
        )
        mock_embedder_factory.create_sparse_document_embedder.return_value = (
            mock_sparse_embedder
        )

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents_with_sparse
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        pipeline = MilvusHybridIndexingPipeline(milvus_config)
        result = pipeline.run()

        mock_dense_embedder.run.assert_called_once()
        mock_sparse_embedder.run.assert_called_once()
        mock_db_instance.create_collection.assert_called_once_with(
            collection_name="test_hybrid",
            dimension=384,
            use_sparse=True,
            recreate=False,
        )
        assert result["documents_indexed"] == 3

    def test_invalid_config_raises(self) -> None:
        """Test that invalid config raises ValueError."""
        from vectordb.haystack.hybrid_indexing.indexing.milvus import (
            MilvusHybridIndexingPipeline,
        )

        invalid_config: dict[str, Any] = {"embeddings": {"model": "test"}}

        with pytest.raises(ValueError, match="milvus"):
            MilvusHybridIndexingPipeline(invalid_config)


class TestMilvusHybridSearch:
    """Tests for MilvusHybridSearchPipeline."""

    @patch("vectordb.haystack.hybrid_indexing.search.milvus.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.search.milvus.MilvusVectorDB")
    def test_init_loads_config(
        self,
        mock_milvus_db: MagicMock,
        mock_embedder_factory: MagicMock,
        milvus_config: dict[str, Any],
    ) -> None:
        """Test that __init__ loads and validates configuration."""
        from vectordb.haystack.hybrid_indexing.search.milvus import (
            MilvusHybridSearchPipeline,
        )

        mock_embedder_factory.create_text_embedder.return_value = MagicMock()
        mock_embedder_factory.create_sparse_text_embedder.return_value = MagicMock()

        pipeline = MilvusHybridSearchPipeline(milvus_config)

        assert pipeline.config == milvus_config
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.ranker_type == "rrf"
        mock_milvus_db.assert_called_once_with(uri="http://localhost:19530", token="")
        mock_embedder_factory.create_text_embedder.assert_called_once()

    @patch("vectordb.haystack.hybrid_indexing.search.milvus.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.search.milvus.MilvusVectorDB")
    def test_run_calls_search(
        self,
        mock_milvus_db: MagicMock,
        mock_embedder_factory: MagicMock,
        milvus_config: dict[str, Any],
        sample_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test that run() embeds query and calls search."""
        from vectordb.haystack.hybrid_indexing.search.milvus import (
            MilvusHybridSearchPipeline,
        )

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = sample_documents
        mock_milvus_db.return_value = mock_db_instance

        mock_text_embedder = MagicMock()
        mock_text_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder
        mock_embedder_factory.create_sparse_text_embedder.return_value = None

        pipeline = MilvusHybridSearchPipeline(milvus_config)
        result = pipeline.run(query="What is machine learning?", top_k=5)

        mock_text_embedder.run.assert_called_once_with(text="What is machine learning?")
        mock_db_instance.search.assert_called_once_with(
            query_embedding=sample_embedding,
            query_sparse_embedding=None,
            top_k=5,
            collection_name="test_hybrid",
            filter=None,
            ranker_type="rrf",
        )
        assert result["documents"] == sample_documents
        assert result["query"] == "What is machine learning?"
        assert result["db"] == "milvus"

    @patch("vectordb.haystack.hybrid_indexing.search.milvus.EmbedderFactory")
    @patch("vectordb.haystack.hybrid_indexing.search.milvus.MilvusVectorDB")
    def test_run_returns_documents(
        self,
        mock_milvus_db: MagicMock,
        mock_embedder_factory: MagicMock,
        milvus_config: dict[str, Any],
        sample_documents: list[Document],
        sample_embedding: list[float],
        sample_sparse_embedding: Any,
    ) -> None:
        """Test that run() returns documents with hybrid search."""
        from vectordb.haystack.hybrid_indexing.search.milvus import (
            MilvusHybridSearchPipeline,
        )

        mock_db_instance = MagicMock()
        mock_db_instance.search.return_value = sample_documents[:2]
        mock_milvus_db.return_value = mock_db_instance

        mock_text_embedder = MagicMock()
        mock_text_embedder.run.return_value = {"embedding": sample_embedding}
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.run.return_value = {
            "sparse_embedding": sample_sparse_embedding
        }

        mock_embedder_factory.create_text_embedder.return_value = mock_text_embedder
        mock_embedder_factory.create_sparse_text_embedder.return_value = (
            mock_sparse_embedder
        )

        pipeline = MilvusHybridSearchPipeline(milvus_config)
        result = pipeline.run(query="neural networks", top_k=2)

        mock_sparse_embedder.run.assert_called_once_with(text="neural networks")
        assert len(result["documents"]) == 2
        assert result["db"] == "milvus"

    def test_invalid_config_raises(self) -> None:
        """Test that invalid config raises ValueError."""
        from vectordb.haystack.hybrid_indexing.search.milvus import (
            MilvusHybridSearchPipeline,
        )

        invalid_config: dict[str, Any] = {"embeddings": {"model": "test"}}

        with pytest.raises(ValueError, match="milvus"):
            MilvusHybridSearchPipeline(invalid_config)
