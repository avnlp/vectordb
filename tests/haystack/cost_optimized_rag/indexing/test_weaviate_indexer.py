"""Tests for WeaviateIndexingPipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer import (
    WeaviateIndexingPipeline,
)


@pytest.fixture
def weaviate_config_dict(base_config_dict: dict[str, Any]) -> dict[str, Any]:
    """Weaviate-specific configuration."""
    config = base_config_dict.copy()
    config["weaviate"] = {
        "host": "localhost",
        "port": 8080,
        "api_key": "",
    }
    return config


@pytest.fixture
def weaviate_config_with_api_key(
    weaviate_config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Weaviate config with API key."""
    config = weaviate_config_dict.copy()
    config["weaviate"]["api_key"] = "test-api-key"
    return config


class TestWeaviateIndexingPipelineInit:
    """Tests for WeaviateIndexingPipeline initialization."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_init_success(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test successful initialization."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = WeaviateIndexingPipeline(config_path)

        assert pipeline.config is not None
        mock_weaviate.Client.assert_called_once()
        mock_embedder_cls.return_value.warm_up.assert_called_once()

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_init_with_api_key(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_with_api_key: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test initialization with API key."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_with_api_key)
        mock_client = MagicMock()
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        WeaviateIndexingPipeline(config_path)

        # Verify auth was passed
        call_kwargs = mock_weaviate.Client.call_args.kwargs
        assert call_kwargs["auth_client_secret"] is not None

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_init_missing_weaviate_config(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_create_logger: MagicMock,
        base_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test initialization fails when weaviate config is missing."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**base_config_dict)

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        with pytest.raises(ValueError, match="Weaviate configuration is missing"):
            WeaviateIndexingPipeline(config_path)


class TestWeaviateIndexingPipelineEnsureClass:
    """Tests for _ensure_class_exists method."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_deletes_existing_class(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that existing class is deleted."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        WeaviateIndexingPipeline(config_path)

        mock_client.schema.delete_class.assert_called_once_with("test_collection")
        mock_client.schema.create_class.assert_called_once()

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_creates_class_with_correct_schema(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test class is created with correct schema."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        WeaviateIndexingPipeline(config_path)

        call_args = mock_client.schema.create_class.call_args[0][0]
        assert call_args["class"] == "test_collection"
        assert call_args["vectorizer"] == "none"
        assert len(call_args["properties"]) == 2  # content and metadata


class TestWeaviateIndexingPipelineRun:
    """Tests for WeaviateIndexingPipeline run method."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_documents_from_config"
    )
    def test_run_success(
        self,
        mock_load_docs: MagicMock,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        sample_documents: list[Document],
        tmp_path: Path,
    ) -> None:
        """Test successful run with documents."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_client.batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_client.batch.__exit__ = MagicMock(return_value=False)
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

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

        pipeline = WeaviateIndexingPipeline(config_path)
        pipeline.run()

        mock_load_docs.assert_called_once()
        assert mock_batch.add_data_object.call_count == 3


class TestWeaviateIndexingPipelineInsertDocuments:
    """Tests for _insert_documents method."""

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_insert_documents_skips_no_embedding(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test that documents without embeddings are skipped."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_client.batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_client.batch.__exit__ = MagicMock(return_value=False)
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = WeaviateIndexingPipeline(config_path)

        docs = [
            Document(id="doc1", content="Has embedding", embedding=[0.1] * 384),
            Document(id="doc2", content="No embedding", embedding=None),
        ]

        pipeline._insert_documents(docs)

        # Only one document should be added
        assert mock_batch.add_data_object.call_count == 1

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_insert_documents_with_metadata_as_json(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test metadata is serialized as JSON."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_client.batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_client.batch.__exit__ = MagicMock(return_value=False)
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = WeaviateIndexingPipeline(config_path)

        docs = [
            Document(
                id="doc1",
                content="Test content",
                meta={"source": "test", "count": 42},
                embedding=[0.1] * 384,
            )
        ]

        pipeline._insert_documents(docs)

        call_kwargs = mock_batch.add_data_object.call_args.kwargs
        assert call_kwargs["data_object"]["content"] == "Test content"
        # Metadata should be JSON string
        import json

        metadata = json.loads(call_kwargs["data_object"]["metadata"])
        assert metadata["source"] == "test"
        assert metadata["count"] == 42

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_insert_documents_uses_batch(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test documents are inserted using batch context manager."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        weaviate_config_dict["embeddings"]["batch_size"] = 2
        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_client.batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_client.batch.__exit__ = MagicMock(return_value=False)
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = WeaviateIndexingPipeline(config_path)

        docs = [
            Document(id=f"doc{i}", content=f"Content {i}", embedding=[0.1] * 384)
            for i in range(5)
        ]

        pipeline._insert_documents(docs)

        # All 5 documents should be added via batch
        assert mock_batch.add_data_object.call_count == 5

    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.create_logger"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.weaviate")
    @patch(
        "vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.SentenceTransformersDocumentEmbedder"
    )
    @patch("vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer.load_config")
    def test_insert_documents_includes_uuid_and_vector(
        self,
        mock_load_config: MagicMock,
        mock_embedder_cls: MagicMock,
        mock_weaviate: MagicMock,
        mock_create_logger: MagicMock,
        weaviate_config_dict: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """Test document ID is used as UUID and vector is passed."""
        from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig

        mock_load_config.return_value = RAGConfig(**weaviate_config_dict)
        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_client.batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_client.batch.__exit__ = MagicMock(return_value=False)
        mock_client.schema.delete_class.side_effect = Exception("Not found")
        mock_weaviate.Client.return_value = mock_client

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy: config")

        pipeline = WeaviateIndexingPipeline(config_path)

        embedding = [0.1] * 384
        docs = [
            Document(id="doc1", content="Content", embedding=embedding),
        ]

        pipeline._insert_documents(docs)

        call_kwargs = mock_batch.add_data_object.call_args.kwargs
        assert call_kwargs["uuid"] == "doc1"
        assert call_kwargs["vector"] == embedding
        assert call_kwargs["class_name"] == "test_collection"
