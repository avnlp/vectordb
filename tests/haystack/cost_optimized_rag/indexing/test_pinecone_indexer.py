"""Tests for PineconeIndexingPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer import (
    PineconeIndexingPipeline,
)


@pytest.fixture
def pinecone_config_dict(base_config_dict: dict[str, Any]) -> dict[str, Any]:
    """Pinecone-specific configuration."""
    config = base_config_dict.copy()
    config["pinecone"] = {
        "api_key": "test-api-key",
        "environment": "us-west4-gcp",
    }
    return config


class TestPineconeIndexingPipelineInit:
    """Tests for PineconeIndexingPipeline initialization."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    def test_init_success(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_pinecone_cls: MagicMock,
        pinecone_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test successful initialization."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**pinecone_config_dict)

        # Mock list_indexes to return existing indexes
        mock_pc = MagicMock()
        mock_index_info = MagicMock()
        mock_index_info.name = "test_collection"
        mock_pc.list_indexes.return_value.indexes = [mock_index_info]
        mock_pinecone_cls.return_value = mock_pc

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = PineconeIndexingPipeline(config_path)

        assert pipeline.config is not None
        mock_pinecone_cls.assert_called_once_with(api_key="test-api-key")
        mock_embedder_cls.return_value.warm_up.assert_called_once()

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    def test_init_creates_index_if_not_exists(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_pinecone_cls: MagicMock,
        pinecone_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test index creation when it doesn't exist."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**pinecone_config_dict)

        mock_pc = MagicMock()
        mock_pc.list_indexes.return_value.indexes = []  # No existing indexes
        mock_pinecone_cls.return_value = mock_pc

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        PineconeIndexingPipeline(config_path)

        mock_pc.create_index.assert_called_once()
        call_kwargs = mock_pc.create_index.call_args.kwargs
        assert call_kwargs["name"] == "test_collection"
        assert call_kwargs["dimension"] == 384
        assert call_kwargs["metric"] == "cosine"

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    def test_init_missing_pinecone_config(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test initialization fails when pinecone config is missing."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**base_config_dict)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        with pytest.raises(ValueError, match="Pinecone configuration is missing"):
            PineconeIndexingPipeline(config_path)


class TestPineconeIndexingPipelineRun:
    """Tests for PineconeIndexingPipeline run method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_documents_from_config"
    )
    def test_run_success(
        self,
        mock_load_docs: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_pinecone_cls: MagicMock,
        pinecone_config_dict: dict[str, Any],
        sample_documents: list[Document],
        tmp_path: Path,
    ) -> None:
        """Test successful run with documents."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**pinecone_config_dict)

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.list_indexes.return_value.indexes = []
        mock_pc.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_pc

        # Add embeddings to documents
        embedded_docs = []
        for doc in sample_documents:
            doc_with_embedding = Document(
                id=doc.id,
                content=doc.content,
                meta=doc.meta,
                embedding=[0.1] * 384,
            )
            embedded_docs.append(doc_with_embedding)

        mock_load_docs.return_value = sample_documents
        mock_embedder_cls.return_value.run.return_value = {"documents": embedded_docs}

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = PineconeIndexingPipeline(config_path)
        pipeline.run()

        mock_load_docs.assert_called_once()
        mock_index.upsert.assert_called()


class TestPineconeIndexingPipelineUpsertDocuments:
    """Tests for _upsert_documents method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    def test_upsert_documents_skips_no_embedding(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_pinecone_cls: MagicMock,
        pinecone_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that documents without embeddings are skipped."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**pinecone_config_dict)

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.list_indexes.return_value.indexes = []
        mock_pc.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_pc

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = PineconeIndexingPipeline(config_path)

        docs = [
            Document(id="doc1", content="Has embedding", embedding=[0.1] * 384),
            Document(id="doc2", content="No embedding", embedding=None),
        ]

        pipeline._upsert_documents(docs)

        # Only one document should be upserted
        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        assert len(vectors) == 1
        assert vectors[0]["id"] == "doc1"

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    def test_upsert_documents_batching(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_pinecone_cls: MagicMock,
        pinecone_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test documents are upserted in batches."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        pinecone_config_dict["embeddings"]["batch_size"] = 2
        mock_load_config.return_value = RAGConfig(**pinecone_config_dict)

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.list_indexes.return_value.indexes = []
        mock_pc.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_pc

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = PineconeIndexingPipeline(config_path)

        docs = [
            Document(id=f"doc{i}", content=f"Content {i}", embedding=[0.1] * 384)
            for i in range(5)
        ]

        pipeline._upsert_documents(docs)

        # 3 batches for 5 docs with batch_size=2
        assert mock_index.upsert.call_count == 3

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    def test_upsert_documents_with_metadata(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_pinecone_cls: MagicMock,
        pinecone_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test metadata is included in upsert."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**pinecone_config_dict)

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.list_indexes.return_value.indexes = []
        mock_pc.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_pc

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = PineconeIndexingPipeline(config_path)

        docs = [
            Document(
                id="doc1",
                content="Test content",
                meta={"source": "test"},
                embedding=[0.1] * 384,
            )
        ]

        pipeline._upsert_documents(docs)

        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs["vectors"]
        assert vectors[0]["metadata"]["content"] == "Test content"
        assert vectors[0]["metadata"]["source"] == "test"

    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.Pinecone")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer.load_config")
    def test_upsert_uses_namespace(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_pinecone_cls: MagicMock,
        pinecone_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test namespace is used in upsert."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**pinecone_config_dict)

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.list_indexes.return_value.indexes = []
        mock_pc.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_pc

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = PineconeIndexingPipeline(config_path)

        docs = [
            Document(id="doc1", content="Content", embedding=[0.1] * 384),
        ]

        pipeline._upsert_documents(docs)

        call_args = mock_index.upsert.call_args
        assert call_args.kwargs["namespace"] == "test_collection"
