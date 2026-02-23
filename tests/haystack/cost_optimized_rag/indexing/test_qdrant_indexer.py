"""Tests for QdrantIndexingPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer import (
    QdrantIndexingPipeline,
)


@pytest.fixture
def qdrant_config_dict(base_config_dict: dict[str, Any]) -> dict[str, Any]:
    """Qdrant-specific configuration."""
    config = base_config_dict.copy()
    config["qdrant"] = {
        "host": "localhost",
        "port": 6333,
        "api_key": "",
        "https": False,
    }
    config["indexing"]["payload_indexes"] = []
    return config


@pytest.fixture
def qdrant_config_with_payload_indexes(
    qdrant_config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Qdrant config with payload indexes."""
    config = qdrant_config_dict.copy()
    config["indexing"]["payload_indexes"] = [
        {"field": "source", "schema_type": "keyword"},
        {"field": "count", "schema_type": "integer"},
    ]
    return config


class TestQdrantIndexingPipelineInit:
    """Tests for QdrantIndexingPipeline initialization."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_init_success(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test successful initialization."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**qdrant_config_dict)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_qdrant_client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = QdrantIndexingPipeline(config_path)

        assert pipeline.config is not None
        mock_qdrant_client.assert_called_once_with(
            host="localhost", port=6333, api_key=None, https=False
        )
        mock_embedder_cls.return_value.warm_up.assert_called_once()

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_init_missing_qdrant_config(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_create_logger: MagicMock,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test initialization fails when qdrant config is missing."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**base_config_dict)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        with pytest.raises(ValueError, match="Qdrant configuration is missing"):
            QdrantIndexingPipeline(config_path)


class TestQdrantIndexingPipelineEnsureCollection:
    """Tests for _ensure_collection_exists method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_recreates_existing_collection(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that existing collection is deleted and recreated."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**qdrant_config_dict)
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()  # Collection exists
        mock_qdrant_client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        QdrantIndexingPipeline(config_path)

        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_client.create_collection.assert_called_once()

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_creates_collection_with_correct_distance(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test collection is created with correct distance metric."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**qdrant_config_dict)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_qdrant_client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        QdrantIndexingPipeline(config_path)

        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_creates_payload_indexes(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_with_payload_indexes: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test payload indexes are created."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**qdrant_config_with_payload_indexes)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_qdrant_client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        QdrantIndexingPipeline(config_path)

        # Should create 2 payload indexes
        assert mock_client.create_payload_index.call_count == 2


class TestQdrantIndexingPipelineRun:
    """Tests for QdrantIndexingPipeline run method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_documents_from_config"
    )
    def test_run_success(
        self,
        mock_load_docs: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_dict: dict[str, Any],
        sample_documents: list[Document],
        tmp_path: Path,
    ) -> None:
        """Test successful run with documents."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**qdrant_config_dict)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_qdrant_client.return_value = mock_client

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

        pipeline = QdrantIndexingPipeline(config_path)
        pipeline.run()

        mock_load_docs.assert_called_once()
        mock_client.upsert.assert_called()


class TestQdrantIndexingPipelineUpsertDocuments:
    """Tests for _upsert_documents method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.PointStruct")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_upsert_documents_skips_no_embedding(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_point_struct: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that documents without embeddings are skipped."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**qdrant_config_dict)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_qdrant_client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = QdrantIndexingPipeline(config_path)

        docs = [
            Document(id="doc1", content="Has embedding", embedding=[0.1] * 384),
            Document(id="doc2", content="No embedding", embedding=None),
        ]

        pipeline._upsert_documents(docs)

        # Only one PointStruct should be created
        assert mock_point_struct.call_count == 1

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_upsert_documents_batching(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test documents are upserted in batches."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        qdrant_config_dict["embeddings"]["batch_size"] = 2
        mock_load_config.return_value = RAGConfig(**qdrant_config_dict)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_qdrant_client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = QdrantIndexingPipeline(config_path)

        docs = [
            Document(id=f"doc{i}", content=f"Content {i}", embedding=[0.1] * 384)
            for i in range(5)
        ]

        pipeline._upsert_documents(docs)

        # 3 batches for 5 docs with batch_size=2
        assert mock_client.upsert.call_count == 3

    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.create_logger")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.QdrantClient")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer.load_config")
    def test_upsert_documents_includes_payload(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_qdrant_client: MagicMock,
        mock_create_logger: MagicMock,
        qdrant_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test payload is included with content and metadata."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**qdrant_config_dict)
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_qdrant_client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = QdrantIndexingPipeline(config_path)

        docs = [
            Document(
                id="doc1",
                content="Test content",
                meta={"source": "test"},
                embedding=[0.1] * 384,
            )
        ]

        pipeline._upsert_documents(docs)

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"
