"""Tests for Chroma namespace pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.namespaces.chroma_namespaces import ChromaNamespacePipeline
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
            name: "test-chroma"
        chroma:
            persist_directory: "./chroma_test_data"
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
def mock_chroma_db():
    """Create a mock ChromaVectorDB."""
    mock_db = MagicMock()
    mock_db.create_index = MagicMock()
    mock_db.create_collection = MagicMock()
    mock_db.delete_collection = MagicMock()
    mock_db.list_collections = MagicMock(return_value=["test_ns_ns1", "test_ns_ns2"])
    mock_db.query = MagicMock(return_value=[])
    mock_db.upsert = MagicMock(return_value=3)
    mock_db._get_collection.return_value.count.return_value = 0
    return mock_db


@pytest.fixture
def mock_embedders():
    """Create mock embedders."""
    mock_doc_embedder = MagicMock()
    mock_doc_embedder.run = MagicMock(return_value={"documents": []})

    mock_text_embedder = MagicMock()
    mock_text_embedder.run = MagicMock(return_value={"embedding": [0.1] * 1024})

    return mock_doc_embedder, mock_text_embedder


class TestChromaNamespaces:
    """Test suite for ChromaNamespacePipeline."""

    def test_initialization(self, tmp_path):
        """Test initialization of ChromaNamespacePipeline."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-chroma"
            chroma:
                persist_directory: "./chroma_test_data"
            embedding:
                model: "Qwen/Qwen3-Embedding-0.6B"
                dimension: 1024
        """)

        # Mock ChromaVectorDB to avoid actual database connection
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db:
            mock_instance = MagicMock()
            mock_instance.create_index = MagicMock()
            mock_db.return_value = mock_instance

            # Pipeline initializes ChromaVectorDB and calls create_index
            pipeline = ChromaNamespacePipeline(str(config_path))
            assert pipeline is not None
            assert pipeline._db is mock_instance
            pipeline.close()

    def test_isolation_strategy(self, tmp_path):
        """Test that isolation strategy is correctly set."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-chroma"
            chroma:
                persist_directory: "./chroma_test_data"
        """)

        # We'll just check the class attribute without connecting
        assert ChromaNamespacePipeline.ISOLATION_STRATEGY.name == "COLLECTION"


class TestChromaNamespacePipelineProperties:
    """Test property accessors for ChromaNamespacePipeline."""

    def test_db_property(self, sample_config, mock_chroma_db):
        """Test db property returns the database instance."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            db = pipeline.db

            assert db is mock_chroma_db
            pipeline.close()

    def test_db_property_raises_when_none(self, sample_config, mock_chroma_db):
        """Test db property raises RuntimeError when db is None."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            pipeline._db = None

            with pytest.raises(RuntimeError, match="Not connected to Chroma"):
                _ = pipeline.db

    def test_logger_property(self, sample_config, mock_chroma_db):
        """Test logger property returns a logger instance."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            logger = pipeline.logger

            assert logger is not None
            assert logger.name == "test-chroma"
            pipeline.close()

    def test_logger_property_default_name(self, tmp_path, mock_chroma_db):
        """Test logger property uses default name when not in config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            chroma:
                persist_directory: "./chroma_test_data"
        """)

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(str(config_path))
            logger = pipeline.logger

            assert logger.name == "chroma_namespaces"
            pipeline.close()

    def test_logger_caching(self, sample_config, mock_chroma_db):
        """Test logger property caches the logger instance."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            logger1 = pipeline.logger
            logger2 = pipeline.logger

            assert logger1 is logger2
            pipeline.close()


class TestChromaNamespacePipelineContextManager:
    """Test context manager for ChromaNamespacePipeline."""

    def test_context_manager_enter(self, sample_config, mock_chroma_db):
        """Test __enter__ returns the pipeline instance."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with ChromaNamespacePipeline(sample_config) as pipeline:
                assert isinstance(pipeline, ChromaNamespacePipeline)

    def test_context_manager_exit(self, sample_config, mock_chroma_db):
        """Test __exit__ closes the connection."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = None
            with ChromaNamespacePipeline(sample_config) as p:
                pipeline = p

            assert pipeline._db is None

    def test_context_manager_exit_with_exception(self, sample_config, mock_chroma_db):
        """Test __exit__ handles exceptions properly."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = None
            try:
                with ChromaNamespacePipeline(sample_config) as p:
                    pipeline = p
                    raise ValueError("Test exception")
            except ValueError:
                assert pipeline is not None

            assert pipeline._db is None


class TestChromaNamespacePipelineNamespaceOperations:
    """Test namespace operations for ChromaNamespacePipeline."""

    def test_create_namespace(self, sample_config, mock_chroma_db):
        """Test create_namespace creates a collection with proper naming."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            result = pipeline.create_namespace("test_ns")

            assert result.success is True
            assert result.namespace == "test_ns"
            assert result.operation == "create"
            assert "test_ns_test_ns" in result.message
            mock_chroma_db.create_collection.assert_called_once_with(
                collection_name="test_ns_test_ns"
            )
            pipeline.close()

    def test_delete_namespace(self, sample_config, mock_chroma_db):
        """Test delete_namespace deletes the collection."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            result = pipeline.delete_namespace("test_ns")

            assert result.success is True
            assert result.namespace == "test_ns"
            assert result.operation == "delete"
            mock_chroma_db.delete_collection.assert_called_once_with(
                collection_name="test_ns_test_ns"
            )
            pipeline.close()

    def test_list_namespaces(self, sample_config, mock_chroma_db):
        """Test list_namespaces returns namespaces with prefix filtering."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            namespaces = pipeline.list_namespaces()

            assert "ns1" in namespaces
            assert "ns2" in namespaces
            mock_chroma_db.list_collections.assert_called()
            pipeline.close()

    def test_list_namespaces_empty(self, sample_config, mock_chroma_db):
        """Test list_namespaces with no matching collections."""
        mock_chroma_db.list_collections.return_value = ["other_collection"]

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            namespaces = pipeline.list_namespaces()

            assert namespaces == []
            pipeline.close()

    def test_namespace_exists_true(self, sample_config, mock_chroma_db):
        """Test namespace_exists returns True when collection exists."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            exists = pipeline.namespace_exists("ns1")

            assert exists is True
            pipeline.close()

    def test_namespace_exists_false(self, sample_config, mock_chroma_db):
        """Test namespace_exists returns False when collection doesn't exist."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            exists = pipeline.namespace_exists("nonexistent")

            assert exists is False
            pipeline.close()

    def test_get_namespace_stats(self, sample_config, mock_chroma_db):
        """Test get_namespace_stats returns correct stats."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_chroma_db._get_collection.return_value = mock_collection

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            stats = pipeline.get_namespace_stats("test_ns")

            assert isinstance(stats, NamespaceStats)
            assert stats.namespace == "test_ns"
            assert stats.document_count == 2
            assert stats.vector_count == 2
            assert stats.status == TenantStatus.ACTIVE
            pipeline.close()

    def test_get_namespace_stats_empty(self, sample_config, mock_chroma_db):
        """Test get_namespace_stats with empty collection."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma_db._get_collection.return_value = mock_collection

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            stats = pipeline.get_namespace_stats("test_ns")

            assert stats.document_count == 0
            assert stats.status == TenantStatus.UNKNOWN
            pipeline.close()


class TestChromaNamespacePipelineIndexing:
    """Test document indexing for ChromaNamespacePipeline."""

    def test_index_documents(
        self, sample_config, mock_chroma_db, mock_embedders, sample_documents
    ):
        """Test index_documents with embedding and upserting."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)
                # Ensure namespace exists
                mock_chroma_db.list_collections.return_value = ["test_ns_test_ns"]

                result = pipeline.index_documents(sample_documents, "test_ns")

                assert result.success is True
                assert result.namespace == "test_ns"
                assert result.operation == "index"
                assert result.data["count"] == 3
                mock_doc_embedder.run.assert_called_once()
                mock_chroma_db.upsert.assert_called_once()
                pipeline.close()

    def test_index_documents_creates_namespace(
        self, sample_config, mock_chroma_db, mock_embedders, sample_documents
    ):
        """Test index_documents creates namespace if it doesn't exist."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)
                # Namespace doesn't exist initially
                mock_chroma_db.list_collections.return_value = []

                result = pipeline.index_documents(sample_documents, "test_ns")

                # Should create namespace first
                mock_chroma_db.create_collection.assert_called_once_with(
                    collection_name="test_ns_test_ns"
                )
                assert result.success is True
                pipeline.close()

    def test_index_documents_empty_list(self, sample_config, mock_chroma_db):
        """Test index_documents with empty document list."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            result = pipeline.index_documents([], "test_ns")

            assert result.success is True
            assert result.data["count"] == 0
            assert "No documents to index" in result.message
            mock_chroma_db.upsert.assert_not_called()
            pipeline.close()

    def test_index_documents_adds_namespace_metadata(
        self, sample_config, mock_chroma_db, mock_embedders, sample_documents
    ):
        """Test index_documents adds namespace to document metadata."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)
                mock_chroma_db.list_collections.return_value = ["test_ns_test_ns"]

                pipeline.index_documents(sample_documents, "test_ns")

                # Check that namespace was added to metadata
                for doc in sample_documents:
                    assert doc.meta.get("namespace") == "test_ns"
                pipeline.close()

    def test_index_documents_no_meta(
        self, sample_config, mock_chroma_db, mock_embedders
    ):
        """Test index_documents handles documents with None meta."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        docs = [Document(content="Test doc", meta=None)]
        mock_doc_embedder.run.return_value = {"documents": docs}

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)
                mock_chroma_db.list_collections.return_value = ["test_ns_test_ns"]

                pipeline.index_documents(docs, "test_ns")

                assert docs[0].meta is not None
                assert docs[0].meta.get("namespace") == "test_ns"
                pipeline.close()

    def test_index_from_config(
        self, sample_config, mock_chroma_db, mock_embedders, sample_documents
    ):
        """Test index_from_config loads documents and indexes them."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.load_documents_from_config",
                    return_value=sample_documents,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)
                mock_chroma_db.list_collections.return_value = ["test_ns_test_ns"]

                result = pipeline.index_from_config("test_ns")

                assert result.success is True
                assert result.data["count"] == 3
                pipeline.close()


class TestChromaNamespacePipelineQuery:
    """Test query methods for ChromaNamespacePipeline."""

    def test_query_namespace(self, sample_config, mock_chroma_db, mock_embedders):
        """Test query_namespace with timing metrics."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [
            Document(content="Result 1", score=0.9),
            Document(content="Result 2", score=0.8),
        ]
        mock_chroma_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)

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
        self, sample_config, mock_chroma_db, mock_embedders
    ):
        """Test query_namespace calls text embedder."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)

                pipeline.query_namespace("test query", "test_ns")

                mock_text_embedder.run.assert_called_once_with(text="test query")
                pipeline.close()

    def test_query_cross_namespace(self, sample_config, mock_chroma_db, mock_embedders):
        """Test query_cross_namespace aggregates results from multiple namespaces."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results_ns1 = [Document(content="NS1 Result", score=0.95)]
        mock_results_ns2 = [Document(content="NS2 Result", score=0.85)]

        def mock_query_side_effect(*, vector, top_k, collection_name):
            if collection_name == "test_ns_ns1":
                return mock_results_ns1
            if collection_name == "test_ns_ns2":
                return mock_results_ns2
            return []

        mock_chroma_db.query.side_effect = mock_query_side_effect

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)

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
        self, sample_config, mock_chroma_db, mock_embedders
    ):
        """Test query_cross_namespace queries all namespaces when none specified."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)

                result = pipeline.query_cross_namespace("test query")

                # Should query all listed namespaces
                assert "ns1" in result.namespace_results
                assert "ns2" in result.namespace_results
                pipeline.close()

    def test_query_cross_namespace_timing_comparison(
        self, sample_config, mock_chroma_db, mock_embedders
    ):
        """Test query_cross_namespace includes timing comparison."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [Document(content="Result", score=0.9)]
        mock_chroma_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            with (
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.chroma_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = ChromaNamespacePipeline(sample_config)

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

    def test_run_method(self, sample_config, mock_chroma_db):
        """Test run method returns pipeline status."""
        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_chroma_db

            pipeline = ChromaNamespacePipeline(sample_config)
            result = pipeline.run()

            assert result["success"] is True
            assert "namespaces" in result
            assert "ns1" in result["namespaces"]
            pipeline.close()


class TestChromaNamespacePipelineExtended:
    """Extended test suite for additional coverage."""

    def test_close_method_sets_db_none(self, tmp_path):
        """Test close method sets _db to None."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-chroma"
            chroma:
                persist_directory: "./chroma_test_data"
        """)

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db:
            mock_instance = MagicMock()
            mock_db.return_value = mock_instance

            pipeline = ChromaNamespacePipeline(str(config_path))
            assert pipeline._db is mock_instance

            pipeline.close()

            assert pipeline._db is None

    def test_context_manager_protocol(self, tmp_path):
        """Test context manager protocol."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-chroma"
            chroma:
                persist_directory: "./chroma_test_data"
        """)

        with patch(
            "vectordb.haystack.namespaces.chroma_namespaces.ChromaVectorDB"
        ) as mock_db:
            mock_instance = MagicMock()
            mock_db.return_value = mock_instance

            with ChromaNamespacePipeline(str(config_path)) as pipeline:
                assert pipeline is not None

            # After context exit, _db should be None
            assert pipeline._db is None
