"""Tests for ChromaIndexingPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer import (
    ChromaIndexingPipeline,
)


@pytest.fixture
def chroma_config_dict(base_config_dict: dict[str, Any]) -> dict[str, Any]:
    """Chroma-specific configuration."""
    config = base_config_dict.copy()
    config["chroma"] = {"path": "/tmp/test_chroma"}
    return config


@pytest.fixture
def mock_chroma_client() -> MagicMock:
    """Mock Chroma client."""
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock embedder that returns documents with embeddings."""
    embedder = MagicMock()
    embedder.warm_up = MagicMock()
    return embedder


class TestChromaIndexingPipelineInit:
    """Tests for ChromaIndexingPipeline initialization."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.chromadb")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_config")
    def test_init_success(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_chromadb: MagicMock,
        mock_create_logger: MagicMock,
        chroma_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test successful initialization."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**chroma_config_dict)
        mock_chromadb.PersistentClient.return_value = MagicMock()

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = ChromaIndexingPipeline(config_path)

        assert pipeline.config is not None
        mock_embedder_cls.return_value.warm_up.assert_called_once()
        mock_chromadb.PersistentClient.assert_called_once()

    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_config")
    def test_init_missing_chroma_config(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_create_logger: MagicMock,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test initialization fails when chroma config is missing."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**base_config_dict)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        with pytest.raises(ValueError, match="Chroma configuration is missing"):
            ChromaIndexingPipeline(config_path)


class TestChromaIndexingPipelineRun:
    """Tests for ChromaIndexingPipeline run method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.chromadb")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_documents_from_config"
    )
    def test_run_success(
        self,
        mock_load_docs: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_chromadb: MagicMock,
        mock_create_logger: MagicMock,
        chroma_config_dict: dict[str, Any],
        sample_documents: list[Document],
        tmp_path: Path,
    ) -> None:
        """Test successful run with documents."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**chroma_config_dict)
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

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

        pipeline = ChromaIndexingPipeline(config_path)
        pipeline.run()

        mock_load_docs.assert_called_once()
        mock_embedder_cls.return_value.run.assert_called_once()
        mock_collection.add.assert_called()

    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.chromadb")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_documents_from_config"
    )
    def test_run_empty_documents(
        self,
        mock_load_docs: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_chromadb: MagicMock,
        mock_create_logger: MagicMock,
        chroma_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test run with no documents."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**chroma_config_dict)
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        mock_load_docs.return_value = []
        mock_embedder_cls.return_value.run.return_value = {"documents": []}

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = ChromaIndexingPipeline(config_path)
        pipeline.run()

        # No documents to add
        mock_collection.add.assert_not_called()


class TestChromaIndexingPipelineAddDocuments:
    """Tests for _add_documents method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.chromadb")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_config")
    def test_add_documents_with_nested_metadata(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_chromadb: MagicMock,
        mock_create_logger: MagicMock,
        chroma_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test adding documents with nested metadata (should be flattened)."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**chroma_config_dict)
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = ChromaIndexingPipeline(config_path)

        # Document with nested metadata
        docs = [
            Document(
                id="doc1",
                content="Test content",
                meta={"simple": "value", "nested": {"key": "value"}},
                embedding=[0.1] * 384,
            )
        ]

        pipeline._add_documents(docs)

        # Verify metadata was flattened (nested dict should be JSON string)
        call_args = mock_collection.add.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
        assert metadatas is not None
        assert metadatas[0]["simple"] == "value"
        assert isinstance(metadatas[0]["nested"], str)  # JSON serialized

    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.chromadb")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_config")
    def test_add_documents_skips_no_embedding(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_chromadb: MagicMock,
        mock_create_logger: MagicMock,
        chroma_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that documents without embeddings are skipped."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**chroma_config_dict)
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = ChromaIndexingPipeline(config_path)

        # Mix of docs with and without embeddings
        docs = [
            Document(id="doc1", content="Has embedding", embedding=[0.1] * 384),
            Document(id="doc2", content="No embedding", embedding=None),
        ]

        pipeline._add_documents(docs)

        call_args = mock_collection.add.call_args
        ids = call_args.kwargs.get("ids") or call_args[1].get("ids")
        assert ids == ["doc1"]

    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.chromadb")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer.load_config")
    def test_add_documents_batching(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_chromadb: MagicMock,
        mock_create_logger: MagicMock,
        chroma_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test documents are added in batches."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        # Set small batch size
        chroma_config_dict["embeddings"]["batch_size"] = 2
        mock_load_config.return_value = RAGConfig(**chroma_config_dict)
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = ChromaIndexingPipeline(config_path)

        # Create 5 documents (should result in 3 batches with batch_size=2)
        docs = [
            Document(id=f"doc{i}", content=f"Content {i}", embedding=[0.1] * 384)
            for i in range(5)
        ]

        pipeline._add_documents(docs)

        # Should have been called 3 times: [2, 2, 1]
        assert mock_collection.add.call_count == 3
