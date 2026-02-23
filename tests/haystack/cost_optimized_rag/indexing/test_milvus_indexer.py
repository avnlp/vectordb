"""Tests for MilvusIndexingPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer import (
    MilvusIndexingPipeline,
)


@pytest.fixture
def milvus_config_dict(base_config_dict: dict[str, Any]) -> dict[str, Any]:
    """Milvus-specific configuration."""
    config = base_config_dict.copy()
    config["milvus"] = {"host": "localhost", "port": 19530}
    config["indexing"]["partitions"] = {"enabled": False, "values": []}
    return config


@pytest.fixture
def milvus_config_with_partitions(
    milvus_config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Milvus config with partitions enabled."""
    config = milvus_config_dict.copy()
    config["indexing"]["partitions"] = {
        "enabled": True,
        "partition_key": "category",
        "values": ["cat1", "cat2"],
    }
    return config


class TestMilvusIndexingPipelineInit:
    """Tests for MilvusIndexingPipeline initialization."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    def test_init_success(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_collection_cls: MagicMock,
        mock_connections: MagicMock,
        milvus_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test successful initialization."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**milvus_config_dict)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = MilvusIndexingPipeline(config_path)

        assert pipeline.config is not None
        mock_connections.connect.assert_called_once_with(
            alias="default", host="localhost", port=19530
        )
        mock_embedder_cls.return_value.warm_up.assert_called_once()

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    def test_init_missing_milvus_config(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test initialization fails when milvus config is missing."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**base_config_dict)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        with pytest.raises(ValueError, match="Milvus configuration is missing"):
            MilvusIndexingPipeline(config_path)

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    def test_init_with_partitions(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_collection_cls: MagicMock,
        mock_connections: MagicMock,
        milvus_config_with_partitions: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test initialization with partitions."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**milvus_config_with_partitions)
        mock_collection = MagicMock()
        mock_collection_cls.return_value = mock_collection

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        MilvusIndexingPipeline(config_path)

        # Should create partitions
        assert mock_collection.create_partition.call_count == 2


class TestMilvusIndexingPipelineRun:
    """Tests for MilvusIndexingPipeline run method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_documents_from_config"
    )
    def test_run_success(
        self,
        mock_load_docs: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_collection_cls: MagicMock,
        mock_connections: MagicMock,
        milvus_config_dict: dict[str, Any],
        sample_documents: list[Document],
        tmp_path: Path,
    ) -> None:
        """Test successful run with documents."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**milvus_config_dict)
        mock_collection = MagicMock()
        mock_collection_cls.return_value = mock_collection

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

        pipeline = MilvusIndexingPipeline(config_path)
        pipeline.run()

        mock_load_docs.assert_called_once()
        mock_collection.insert.assert_called()
        mock_collection.create_index.assert_called_once()
        mock_collection.load.assert_called_once()


class TestMilvusIndexingPipelineInsertDocuments:
    """Tests for _insert_documents method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    def test_insert_documents_skips_no_embedding(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_collection_cls: MagicMock,
        mock_connections: MagicMock,
        milvus_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that documents without embeddings are skipped."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**milvus_config_dict)
        mock_collection = MagicMock()
        mock_collection_cls.return_value = mock_collection

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = MilvusIndexingPipeline(config_path)

        docs = [
            Document(id="doc1", content="Has embedding", embedding=[0.1] * 384),
            Document(id="doc2", content="No embedding", embedding=None),
        ]

        pipeline._insert_documents(docs)

        # Only one document should be inserted
        call_args = mock_collection.insert.call_args[0][0]
        assert len(call_args[0]) == 1  # ids list

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    def test_insert_documents_batching(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_collection_cls: MagicMock,
        mock_connections: MagicMock,
        milvus_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test documents are inserted in batches."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        milvus_config_dict["embeddings"]["batch_size"] = 2
        mock_load_config.return_value = RAGConfig(**milvus_config_dict)
        mock_collection = MagicMock()
        mock_collection_cls.return_value = mock_collection

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = MilvusIndexingPipeline(config_path)

        docs = [
            Document(id=f"doc{i}", content=f"Content {i}", embedding=[0.1] * 384)
            for i in range(5)
        ]

        pipeline._insert_documents(docs)

        # 3 batches for 5 docs with batch_size=2
        assert mock_collection.insert.call_count == 3


class TestMilvusIndexingPipelineEnsureCollection:
    """Tests for _ensure_collection_exists method."""

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.CollectionSchema"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    def test_drops_existing_collection(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_schema: MagicMock,
        mock_collection_cls: MagicMock,
        mock_connections: MagicMock,
        milvus_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that existing collection is dropped."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**milvus_config_dict)

        # First Collection call (for drop) succeeds
        mock_existing_collection = MagicMock()
        # Second Collection call (for create) returns new collection
        mock_new_collection = MagicMock()
        mock_collection_cls.side_effect = [
            mock_existing_collection,
            mock_new_collection,
        ]

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        MilvusIndexingPipeline(config_path)

        mock_existing_collection.drop.assert_called_once()

    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.connections")
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.Collection")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer.load_config")
    def test_handles_nonexistent_collection(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_collection_cls: MagicMock,
        mock_connections: MagicMock,
        milvus_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test handling when collection doesn't exist."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**milvus_config_dict)

        # First call raises exception (collection doesn't exist)
        # Second call creates new collection
        mock_collection = MagicMock()
        mock_collection_cls.side_effect = [Exception("Not found"), mock_collection]

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        # Should not raise
        MilvusIndexingPipeline(config_path)
