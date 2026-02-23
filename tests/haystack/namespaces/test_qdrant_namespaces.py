"""Tests for Qdrant namespace pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.namespaces.qdrant_namespaces import QdrantNamespacePipeline
from vectordb.haystack.namespaces.types import (
    NamespaceStats,
    TenantStatus,
)


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
        pipeline:
            name: "test-qdrant"
        qdrant:
            host: "localhost"
            port: 6333
        collection:
            name: "test_ns"
        embedding:
            model: "Qwen/Qwen3-Embedding-0.6B"
            dimension: 1024
        indexing:
            batch_size: 50
    """)
    return str(config_path)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(content="Document 1", meta={"id": "1"}),
        Document(content="Document 2", meta={"id": "2"}),
        Document(content="Document 3", meta={"id": "3"}),
    ]


@pytest.fixture
def mock_qdrant_db():
    """Create a mock QdrantVectorDB."""
    mock_db = MagicMock()
    mock_db.create_index = MagicMock()
    mock_db.delete = MagicMock()
    mock_db.query = MagicMock(return_value=[])
    mock_db.upsert = MagicMock(return_value=3)
    mock_db.collection_name = "test_collection"
    # Default: scroll returns empty, count returns 0
    mock_db.client.scroll.return_value = ([], None)
    mock_db.client.count.return_value = MagicMock(count=0)
    return mock_db


@pytest.fixture
def mock_embedders():
    """Create mock embedders."""
    mock_doc_embedder = MagicMock()
    mock_doc_embedder.run = MagicMock(return_value={"documents": []})

    mock_text_embedder = MagicMock()
    mock_text_embedder.run = MagicMock(return_value={"embedding": [0.1] * 1024})

    return mock_doc_embedder, mock_text_embedder


class TestQdrantNamespaces:
    """Test suite for QdrantNamespacePipeline."""

    def test_initialization(self, tmp_path):
        """Test initialization of QdrantNamespacePipeline."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-qdrant"
            qdrant:
                host: "localhost"
                port: 6333
            embedding:
                model: "Qwen/Qwen3-Embedding-0.6B"
                dimension: 1024
        """)

        # Mock QdrantVectorDB to avoid actual database connection
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db:
            mock_instance = MagicMock()
            mock_instance.create_index = MagicMock()
            mock_db.return_value = mock_instance

            # Pipeline initializes QdrantVectorDB and calls create_index
            pipeline = QdrantNamespacePipeline(str(config_path))
            assert pipeline is not None
            assert pipeline._db is mock_instance
            pipeline.close()

    def test_isolation_strategy(self, tmp_path):
        """Test that isolation strategy is correctly set."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-qdrant"
            qdrant:
                host: "localhost"
                port: 6333
        """)

        # We'll just check the class attribute without connecting
        assert QdrantNamespacePipeline.ISOLATION_STRATEGY.name == "PAYLOAD_FILTER"


class TestQdrantNamespacePipelineProperties:
    """Test property accessors for QdrantNamespacePipeline."""

    def test_db_property(self, sample_config, mock_qdrant_db):
        """Test db property returns the database instance."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            db = pipeline.db

            assert db is mock_qdrant_db
            pipeline.close()

    def test_db_property_raises_when_none(self, sample_config, mock_qdrant_db):
        """Test db property raises RuntimeError when db is None."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            pipeline._db = None

            with pytest.raises(RuntimeError, match="Not connected to Qdrant"):
                _ = pipeline.db

    def test_logger_property(self, sample_config, mock_qdrant_db):
        """Test logger property returns a logger instance."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            logger = pipeline.logger

            assert logger is not None
            assert logger.name == "test-qdrant"
            pipeline.close()

    def test_logger_property_default_name(self, tmp_path, mock_qdrant_db):
        """Test logger property uses default name when not in config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            qdrant:
                host: "localhost"
                port: 6333
        """)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(str(config_path))
            logger = pipeline.logger

            assert logger.name == "qdrant_namespaces"
            pipeline.close()

    def test_logger_caching(self, sample_config, mock_qdrant_db):
        """Test logger property caches the logger instance."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            logger1 = pipeline.logger
            logger2 = pipeline.logger

            assert logger1 is logger2
            pipeline.close()


class TestQdrantNamespacePipelineContextManager:
    """Test context manager for QdrantNamespacePipeline."""

    def test_context_manager_enter(self, sample_config, mock_qdrant_db):
        """Test __enter__ returns the pipeline instance."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with QdrantNamespacePipeline(sample_config) as pipeline:
                assert isinstance(pipeline, QdrantNamespacePipeline)

    def test_context_manager_exit(self, sample_config, mock_qdrant_db):
        """Test __exit__ closes the connection."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = None
            with QdrantNamespacePipeline(sample_config) as p:
                pipeline = p

            assert pipeline._db is None

    def test_context_manager_exit_with_exception(self, sample_config, mock_qdrant_db):
        """Test __exit__ handles exceptions properly."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = None
            try:
                with QdrantNamespacePipeline(sample_config) as p:
                    pipeline = p
                    raise ValueError("Test exception")
            except ValueError:
                assert pipeline is not None

            assert pipeline._db is None


class TestQdrantNamespacePipelineNamespaceOperations:
    """Test namespace operations for QdrantNamespacePipeline."""

    def test_create_namespace(self, sample_config, mock_qdrant_db):
        """Test create_namespace returns success for payload filter strategy."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            result = pipeline.create_namespace("test_ns")

            assert result.success is True
            assert result.namespace == "test_ns"
            assert result.operation == "create"
            assert "auto-created on insert" in result.message
            pipeline.close()

    def test_delete_namespace(self, sample_config, mock_qdrant_db):
        """Test delete_namespace deletes documents with namespace filter."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            result = pipeline.delete_namespace("test_ns")

            assert result.success is True
            assert result.namespace == "test_ns"
            assert result.operation == "delete"
            mock_qdrant_db.delete.assert_called_once_with(
                filters={"namespace": "test_ns"}
            )
            pipeline.close()

    def test_list_namespaces(self, sample_config, mock_qdrant_db):
        """Test list_namespaces returns unique namespace values."""
        mock_records = [
            MagicMock(payload={"namespace": "ns1"}),
            MagicMock(payload={"namespace": "ns2"}),
            MagicMock(payload={"namespace": "ns1"}),  # Duplicate namespace
        ]
        mock_qdrant_db.client.scroll.return_value = (mock_records, None)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            namespaces = pipeline.list_namespaces()

            assert "ns1" in namespaces
            assert "ns2" in namespaces
            assert len(namespaces) == 2  # Unique values only
            mock_qdrant_db.client.scroll.assert_called_once()
            pipeline.close()

    def test_list_namespaces_empty(self, sample_config, mock_qdrant_db):
        """Test list_namespaces with no documents."""
        mock_qdrant_db.client.scroll.return_value = ([], None)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            namespaces = pipeline.list_namespaces()

            assert namespaces == []
            pipeline.close()

    def test_list_namespaces_docs_without_namespace(
        self, sample_config, mock_qdrant_db
    ):
        """Test list_namespaces ignores docs without namespace metadata."""
        mock_records = [
            MagicMock(payload={"namespace": "ns1"}),
            MagicMock(payload={"other": "value"}),  # No namespace
            MagicMock(payload={"namespace": "ns2"}),
        ]
        mock_qdrant_db.client.scroll.return_value = (mock_records, None)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            namespaces = pipeline.list_namespaces()

            assert "ns1" in namespaces
            assert "ns2" in namespaces
            assert "other" not in namespaces
            pipeline.close()

    def test_namespace_exists_true(self, sample_config, mock_qdrant_db):
        """Test namespace_exists returns True when documents exist."""
        mock_qdrant_db.client.count.return_value = MagicMock(count=5)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            exists = pipeline.namespace_exists("test_ns")

            assert exists is True
            mock_qdrant_db.client.count.assert_called_once()
            pipeline.close()

    def test_namespace_exists_false(self, sample_config, mock_qdrant_db):
        """Test namespace_exists returns False when no documents exist."""
        mock_qdrant_db.client.count.return_value = MagicMock(count=0)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            exists = pipeline.namespace_exists("nonexistent")

            assert exists is False
            pipeline.close()

    def test_get_namespace_stats(self, sample_config, mock_qdrant_db):
        """Test get_namespace_stats returns correct stats."""
        mock_qdrant_db.client.count.return_value = MagicMock(count=2)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            stats = pipeline.get_namespace_stats("test_ns")

            assert isinstance(stats, NamespaceStats)
            assert stats.namespace == "test_ns"
            assert stats.document_count == 2
            assert stats.vector_count == 2
            assert stats.status == TenantStatus.ACTIVE
            pipeline.close()

    def test_get_namespace_stats_empty(self, sample_config, mock_qdrant_db):
        """Test get_namespace_stats with empty namespace."""
        mock_qdrant_db.client.count.return_value = MagicMock(count=0)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            stats = pipeline.get_namespace_stats("test_ns")

            assert stats.document_count == 0
            assert stats.status == TenantStatus.UNKNOWN
            pipeline.close()


class TestQdrantNamespacePipelineIndexing:
    """Test document indexing for QdrantNamespacePipeline."""

    def test_index_documents(
        self, sample_config, mock_qdrant_db, mock_embedders, sample_documents
    ):
        """Test index_documents with embedding and upserting."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                result = pipeline.index_documents(sample_documents, "test_ns")

                assert result.success is True
                assert result.namespace == "test_ns"
                assert result.operation == "index"
                assert result.data["count"] == 3
                mock_doc_embedder.run.assert_called_once()
                mock_qdrant_db.upsert.assert_called_once()
                pipeline.close()

    def test_index_documents_empty_list(self, sample_config, mock_qdrant_db):
        """Test index_documents with empty document list."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            result = pipeline.index_documents([], "test_ns")

            assert result.success is True
            assert result.data["count"] == 0
            assert "No documents to index" in result.message
            mock_qdrant_db.upsert.assert_not_called()
            pipeline.close()

    def test_index_documents_adds_namespace_metadata(
        self, sample_config, mock_qdrant_db, mock_embedders, sample_documents
    ):
        """Test index_documents adds namespace to document metadata."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                pipeline.index_documents(sample_documents, "test_ns")

                # Check that namespace was added to metadata
                for doc in sample_documents:
                    assert doc.meta.get("namespace") == "test_ns"
                pipeline.close()

    def test_index_documents_no_meta(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test index_documents handles documents with None meta."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        docs = [Document(content="Test doc", meta=None)]
        mock_doc_embedder.run.return_value = {"documents": docs}

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                pipeline.index_documents(docs, "test_ns")

                assert docs[0].meta is not None
                assert docs[0].meta.get("namespace") == "test_ns"
                pipeline.close()

    def test_index_from_config(
        self, sample_config, mock_qdrant_db, mock_embedders, sample_documents
    ):
        """Test index_from_config loads documents and indexes them."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.load_documents_from_config",
                    return_value=sample_documents,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                result = pipeline.index_from_config("test_ns")

                assert result.success is True
                assert result.data["count"] == 3
                pipeline.close()

    def test_index_documents_with_batch_size(
        self, sample_config, mock_qdrant_db, mock_embedders, sample_documents
    ):
        """Test index_documents uses batch_size from config."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                pipeline.index_documents(sample_documents, "test_ns")

                # Check that upsert was called with correct batch_size from config (50)
                mock_qdrant_db.upsert.assert_called_once()
                call_kwargs = mock_qdrant_db.upsert.call_args[1]
                assert call_kwargs.get("batch_size") == 50
                pipeline.close()


class TestQdrantNamespacePipelineQuery:
    """Test query methods for QdrantNamespacePipeline."""

    def test_query_namespace(self, sample_config, mock_qdrant_db, mock_embedders):
        """Test query_namespace with timing metrics."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [
            Document(content="Result 1", score=0.9),
            Document(content="Result 2", score=0.8),
        ]
        mock_qdrant_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                results = pipeline.query_namespace("test query", "test_ns", top_k=5)

                assert len(results) == 2
                assert results[0].document.content == "Result 1"
                assert results[0].relevance_score == 0.9
                assert results[0].rank == 1
                assert results[0].namespace == "test_ns"
                assert results[0].timing is not None
                assert results[1].timing is None  # Only first result has timing
                pipeline.close()

    def test_query_namespace_embedder_called(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test query_namespace calls text embedder."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                pipeline.query_namespace("test query", "test_ns")

                mock_text_embedder.run.assert_called_once_with(text="test query")
                pipeline.close()

    def test_query_namespace_with_namespace_filter(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test query_namespace applies namespace filter."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                pipeline.query_namespace("test query", "my_namespace", top_k=10)

                # query_namespace calls db.query twice:
                # 1. For the actual search with the query embedding
                # 2. For get_namespace_stats with a dummy query
                assert mock_qdrant_db.query.call_count >= 1
                # Check the first call was the search query
                first_call_kwargs = mock_qdrant_db.query.call_args_list[0][1]
                assert first_call_kwargs.get("filters") == {"namespace": "my_namespace"}
                assert first_call_kwargs.get("top_k") == 10
                pipeline.close()

    def test_query_cross_namespace(self, sample_config, mock_qdrant_db, mock_embedders):
        """Test query_cross_namespace aggregates results from multiple namespaces."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results_ns1 = [Document(content="NS1 Result", score=0.95)]
        mock_results_ns2 = [Document(content="NS2 Result", score=0.85)]

        def mock_query_side_effect(*, vector, top_k, filters):
            if filters.get("namespace") == "ns1":
                return mock_results_ns1
            if filters.get("namespace") == "ns2":
                return mock_results_ns2
            return []

        mock_qdrant_db.query.side_effect = mock_query_side_effect

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                result = pipeline.query_cross_namespace(
                    "test query", namespaces=["ns1", "ns2"], top_k=5
                )

                assert result.query == "test query"
                assert "ns1" in result.namespace_results
                assert "ns2" in result.namespace_results
                assert len(result.timing_comparison) == 2
                assert result.total_time_ms >= 0
                pipeline.close()

    def test_query_cross_namespace_all_namespaces(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test query_cross_namespace queries all namespaces when none specified."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        # Mock list_namespaces via scroll, and get_namespace_stats via count
        mock_records = [
            MagicMock(payload={"namespace": "ns1"}),
            MagicMock(payload={"namespace": "ns2"}),
        ]
        mock_qdrant_db.client.scroll.return_value = (mock_records, None)
        mock_qdrant_db.client.count.return_value = MagicMock(count=1)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                # query_cross_namespace calls:
                # 1. list_namespaces() which uses client.scroll
                # 2. For each namespace, query_namespace() which calls:
                #    - db.query for the search
                #    - client.count for get_namespace_stats
                mock_qdrant_db.query.side_effect = [
                    [Document(content="Result 1")],  # ns1 search query
                    [Document(content="Result 2")],  # ns2 search query
                ]

                result = pipeline.query_cross_namespace("test query")

                # Should query all listed namespaces
                assert "ns1" in result.namespace_results
                assert "ns2" in result.namespace_results
                pipeline.close()

    def test_query_cross_namespace_timing_comparison(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test query_cross_namespace includes timing comparison."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [Document(content="Result", score=0.9)]
        mock_qdrant_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                result = pipeline.query_cross_namespace(
                    "test query", namespaces=["ns1"]
                )

                assert len(result.timing_comparison) == 1
                comparison = result.timing_comparison[0]
                assert comparison.namespace == "ns1"
                assert comparison.result_count == 1
                assert comparison.top_score == 0.9
                assert comparison.timing is not None
                pipeline.close()

    def test_query_cross_namespace_empty_results(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test query_cross_namespace handles empty results."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_qdrant_db.query.return_value = []

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                result = pipeline.query_cross_namespace(
                    "test query", namespaces=["ns1"]
                )

                # When there are no results, no timing comparison is added
                # because the code only adds comparisons when results exist
                assert len(result.timing_comparison) == 0
                assert "ns1" in result.namespace_results
                assert result.namespace_results["ns1"] == []
                pipeline.close()

    def test_query_cross_namespace_with_none_timing(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test query_cross_namespace with None timing for coverage."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [Document(content="Result", score=0.9)]
        mock_qdrant_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                # Mock query_namespace to return results with None timing
                # This triggers the else branch in query_cross_namespace
                with patch.object(pipeline, "query_namespace") as mock_query_ns:
                    from vectordb.haystack.namespaces.types import NamespaceQueryResult

                    mock_query_ns.return_value = [
                        NamespaceQueryResult(
                            document=Document(content="Result", score=0.9),
                            relevance_score=0.9,
                            rank=1,
                            namespace="ns1",
                            timing=None,  # None timing triggers else branch
                        )
                    ]

                    result = pipeline.query_cross_namespace(
                        "test query", namespaces=["ns1"]
                    )

                    assert len(result.timing_comparison) == 1
                    comparison = result.timing_comparison[0]
                    assert comparison.namespace == "ns1"
                    assert comparison.result_count == 1
                    assert comparison.top_score == 0.9
                    # When timing is None, default timing metrics are created
                    assert comparison.timing is not None
                    assert comparison.timing.namespace_lookup_ms == 0.0
                    assert comparison.timing.vector_search_ms == 0.0
                    assert comparison.timing.total_ms == 0.0
                    pipeline.close()


class TestQdrantNamespacePipelineRun:
    """Test run method for QdrantNamespacePipeline."""

    def test_run_method(self, sample_config, mock_qdrant_db):
        """Test run method returns pipeline status."""
        mock_records = [MagicMock(payload={"namespace": "ns1"})]
        mock_qdrant_db.client.scroll.return_value = (mock_records, None)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            result = pipeline.run()

            assert result["success"] is True
            assert "message" in result
            assert "namespaces" in result
            assert "ns1" in result["namespaces"]
            pipeline.close()


class TestQdrantNamespacePipelineClose:
    """Test close method for QdrantNamespacePipeline."""

    def test_close_method(self, sample_config, mock_qdrant_db):
        """Test close method clears the database connection."""
        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)
            assert pipeline._db is not None

            pipeline.close()

            assert pipeline._db is None

    def test_close_logs_info(self, sample_config, mock_qdrant_db, caplog):
        """Test close method logs info message."""
        import logging

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            pipeline = QdrantNamespacePipeline(sample_config)

            with caplog.at_level(logging.INFO):
                pipeline.close()

            assert "Closed Qdrant connection" in caplog.text


class TestQdrantNamespacePipelineInitEmbedders:
    """Test _init_embedders method for QdrantNamespacePipeline."""

    def test_init_embedders_lazy_loading(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test embedders are loaded lazily."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ) as mock_get_doc_embedder,
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ) as mock_get_text_embedder,
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                # Embedders should not be initialized yet
                assert pipeline._doc_embedder is None
                assert pipeline._text_embedder is None

                # Call _init_embedders
                pipeline._init_embedders()

                # Now they should be initialized
                assert pipeline._doc_embedder is mock_doc_embedder
                assert pipeline._text_embedder is mock_text_embedder
                mock_get_doc_embedder.assert_called_once()
                mock_get_text_embedder.assert_called_once()

                pipeline.close()

    def test_init_embedders_caching(
        self, sample_config, mock_qdrant_db, mock_embedders
    ):
        """Test embedders are cached after first initialization."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_qdrant_db

            with (
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ) as mock_get_doc_embedder,
                patch(
                    "vectordb.haystack.namespaces.qdrant_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = QdrantNamespacePipeline(sample_config)

                # Initialize twice
                pipeline._init_embedders()
                pipeline._init_embedders()

                # Should only be called once
                mock_get_doc_embedder.assert_called_once()

                pipeline.close()


class TestQdrantNamespacePipelineExtended:
    """Extended test suite for additional coverage."""

    def test_close_method_sets_db_none(self, tmp_path):
        """Test close method sets _db to None."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-qdrant"
            qdrant:
                url: "http://localhost:6333"
        """)

        with patch(
            "vectordb.haystack.namespaces.qdrant_namespaces.QdrantVectorDB"
        ) as mock_db:
            mock_instance = MagicMock()
            mock_db.return_value = mock_instance

            pipeline = QdrantNamespacePipeline(str(config_path))
            assert pipeline._db is mock_instance

            pipeline.close()

            assert pipeline._db is None
