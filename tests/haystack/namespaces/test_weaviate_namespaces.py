"""Tests for Weaviate namespace pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.namespaces.types import (
    NamespaceStats,
    TenantStatus,
)
from vectordb.haystack.namespaces.weaviate_namespaces import WeaviateNamespacePipeline


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
        pipeline:
            name: "test-weaviate"
        weaviate:
            host: "localhost"
            port: 8080
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
def mock_weaviate_db():
    """Create a mock WeaviateVectorDB."""
    mock_db = MagicMock()
    mock_db.create_index = MagicMock()
    mock_db.create_tenant = MagicMock()
    mock_db.delete_tenant = MagicMock()
    mock_db.list_tenants = MagicMock(return_value=[])
    mock_db.query = MagicMock(return_value=[])
    mock_db.upsert = MagicMock(return_value=3)
    mock_db.collection.aggregate.over_all.return_value.total_count = 0
    return mock_db


@pytest.fixture
def mock_embedders():
    """Create mock embedders."""
    mock_doc_embedder = MagicMock()
    mock_doc_embedder.run = MagicMock(return_value={"documents": []})

    mock_text_embedder = MagicMock()
    mock_text_embedder.run = MagicMock(return_value={"embedding": [0.1] * 1024})

    return mock_doc_embedder, mock_text_embedder


class TestWeaviateNamespaces:
    """Test suite for WeaviateNamespacePipeline."""

    def test_initialization(self, tmp_path):
        """Test initialization of WeaviateNamespacePipeline."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-weaviate"
            weaviate:
                host: "localhost"
                port: 8080
            embedding:
                model: "Qwen/Qwen3-Embedding-0.6B"
                dimension: 1024
        """)

        # Mock WeaviateVectorDB to avoid actual database connection
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db:
            mock_instance = MagicMock()
            mock_instance.create_index = MagicMock()
            mock_db.return_value = mock_instance

            # Pipeline initializes WeaviateVectorDB and calls create_index
            pipeline = WeaviateNamespacePipeline(str(config_path))
            assert pipeline is not None
            assert pipeline._db is mock_instance
            pipeline.close()

    def test_isolation_strategy(self, tmp_path):
        """Test that isolation strategy is correctly set."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-weaviate"
            weaviate:
                host: "localhost"
                port: 8080
        """)

        # We'll just check the class attribute without connecting
        assert WeaviateNamespacePipeline.ISOLATION_STRATEGY.name == "TENANT"


class TestWeaviateNamespacePipelineProperties:
    """Test property accessors for WeaviateNamespacePipeline."""

    def test_db_property(self, sample_config, mock_weaviate_db):
        """Test db property returns the database instance."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            db = pipeline.db

            assert db is mock_weaviate_db
            pipeline.close()

    def test_db_property_raises_when_none(self, sample_config, mock_weaviate_db):
        """Test db property raises RuntimeError when db is None."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            pipeline._db = None

            with pytest.raises(RuntimeError, match="Not connected to Weaviate"):
                _ = pipeline.db

    def test_logger_property(self, sample_config, mock_weaviate_db):
        """Test logger property returns a logger instance."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            logger = pipeline.logger

            assert logger is not None
            assert logger.name == "test-weaviate"
            pipeline.close()

    def test_logger_property_default_name(self, tmp_path, mock_weaviate_db):
        """Test logger property uses default name when not in config."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            weaviate:
                host: "localhost"
                port: 8080
        """)

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(str(config_path))
            logger = pipeline.logger

            assert logger.name == "weaviate_namespaces"
            pipeline.close()

    def test_logger_caching(self, sample_config, mock_weaviate_db):
        """Test logger property caches the logger instance."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            logger1 = pipeline.logger
            logger2 = pipeline.logger

            assert logger1 is logger2
            pipeline.close()


class TestWeaviateNamespacePipelineContextManager:
    """Test context manager for WeaviateNamespacePipeline."""

    def test_context_manager_enter(self, sample_config, mock_weaviate_db):
        """Test __enter__ returns the pipeline instance."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with WeaviateNamespacePipeline(sample_config) as pipeline:
                assert isinstance(pipeline, WeaviateNamespacePipeline)

    def test_context_manager_exit(self, sample_config, mock_weaviate_db):
        """Test __exit__ closes the connection."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = None
            with WeaviateNamespacePipeline(sample_config) as p:
                pipeline = p

            assert pipeline._db is None

    def test_context_manager_exit_with_exception(self, sample_config, mock_weaviate_db):
        """Test __exit__ handles exceptions properly."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = None
            try:
                with WeaviateNamespacePipeline(sample_config) as p:
                    pipeline = p
                    raise ValueError("Test exception")
            except ValueError:
                assert pipeline is not None

            assert pipeline._db is None


class TestWeaviateNamespacePipelineNamespaceOperations:
    """Test namespace operations for WeaviateNamespacePipeline."""

    def test_create_namespace(self, sample_config, mock_weaviate_db):
        """Test create_namespace creates a tenant in Weaviate."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            result = pipeline.create_namespace("test_ns")

            assert result.success is True
            assert result.namespace == "test_ns"
            assert result.operation == "create"
            assert "Created tenant" in result.message
            mock_weaviate_db.create_tenant.assert_called_once_with(tenant="test_ns")
            pipeline.close()

    def test_delete_namespace(self, sample_config, mock_weaviate_db):
        """Test delete_namespace deletes the tenant from Weaviate."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            result = pipeline.delete_namespace("test_ns")

            assert result.success is True
            assert result.namespace == "test_ns"
            assert result.operation == "delete"
            assert "Deleted tenant" in result.message
            mock_weaviate_db.delete_tenant.assert_called_once_with(tenant="test_ns")
            pipeline.close()

    def test_list_namespaces(self, sample_config, mock_weaviate_db):
        """Test list_namespaces returns all tenants."""
        mock_weaviate_db.list_tenants.return_value = ["ns1", "ns2", "ns3"]

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            namespaces = pipeline.list_namespaces()

            assert "ns1" in namespaces
            assert "ns2" in namespaces
            assert "ns3" in namespaces
            assert len(namespaces) == 3
            mock_weaviate_db.list_tenants.assert_called_once()
            pipeline.close()

    def test_list_namespaces_empty(self, sample_config, mock_weaviate_db):
        """Test list_namespaces with no tenants."""
        mock_weaviate_db.list_tenants.return_value = []

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            namespaces = pipeline.list_namespaces()

            assert namespaces == []
            pipeline.close()

    def test_namespace_exists_true(self, sample_config, mock_weaviate_db):
        """Test namespace_exists returns True when tenant exists."""
        mock_weaviate_db.list_tenants.return_value = ["ns1", "ns2"]

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            exists = pipeline.namespace_exists("ns1")

            assert exists is True
            pipeline.close()

    def test_namespace_exists_false(self, sample_config, mock_weaviate_db):
        """Test namespace_exists returns False when tenant doesn't exist."""
        mock_weaviate_db.list_tenants.return_value = ["ns1", "ns2"]

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            exists = pipeline.namespace_exists("nonexistent")

            assert exists is False
            pipeline.close()

    def test_get_namespace_stats(self, sample_config, mock_weaviate_db):
        """Test get_namespace_stats returns correct stats."""
        mock_weaviate_db.collection.aggregate.over_all.return_value.total_count = 2

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            stats = pipeline.get_namespace_stats("test_ns")

            assert isinstance(stats, NamespaceStats)
            assert stats.namespace == "test_ns"
            assert stats.document_count == 2
            assert stats.vector_count == 2
            assert stats.status == TenantStatus.ACTIVE
            pipeline.close()

    def test_get_namespace_stats_empty(self, sample_config, mock_weaviate_db):
        """Test get_namespace_stats with empty namespace."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            stats = pipeline.get_namespace_stats("test_ns")

            assert stats.document_count == 0
            assert stats.status == TenantStatus.UNKNOWN
            pipeline.close()


class TestWeaviateNamespacePipelineIndexing:
    """Test document indexing for WeaviateNamespacePipeline."""

    def test_index_documents(
        self, sample_config, mock_weaviate_db, mock_embedders, sample_documents
    ):
        """Test index_documents with embedding and upserting."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)
                # Ensure namespace exists
                mock_weaviate_db.list_tenants.return_value = ["test_ns"]

                result = pipeline.index_documents(sample_documents, "test_ns")

                assert result.success is True
                assert result.namespace == "test_ns"
                assert result.operation == "index"
                assert result.data["count"] == 3
                mock_doc_embedder.run.assert_called_once()
                mock_weaviate_db.upsert.assert_called_once()
                pipeline.close()

    def test_index_documents_creates_namespace(
        self, sample_config, mock_weaviate_db, mock_embedders, sample_documents
    ):
        """Test index_documents creates namespace if it doesn't exist."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)
                # Namespace doesn't exist initially
                mock_weaviate_db.list_tenants.return_value = []

                result = pipeline.index_documents(sample_documents, "test_ns")

                # Should create namespace first
                mock_weaviate_db.create_tenant.assert_called_once_with(tenant="test_ns")
                assert result.success is True
                pipeline.close()

    def test_index_documents_empty_list(self, sample_config, mock_weaviate_db):
        """Test index_documents with empty document list."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            result = pipeline.index_documents([], "test_ns")

            assert result.success is True
            assert result.data["count"] == 0
            assert "No documents to index" in result.message
            mock_weaviate_db.upsert.assert_not_called()
            pipeline.close()

    def test_index_documents_adds_namespace_metadata(
        self, sample_config, mock_weaviate_db, mock_embedders, sample_documents
    ):
        """Test index_documents adds namespace to document metadata."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)
                mock_weaviate_db.list_tenants.return_value = ["test_ns"]

                pipeline.index_documents(sample_documents, "test_ns")

                # Check that namespace was added to metadata
                for doc in sample_documents:
                    assert doc.meta.get("namespace") == "test_ns"
                pipeline.close()

    def test_index_documents_no_meta(
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test index_documents handles documents with None meta."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        docs = [Document(content="Test doc", meta=None)]
        mock_doc_embedder.run.return_value = {"documents": docs}

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)
                mock_weaviate_db.list_tenants.return_value = ["test_ns"]

                pipeline.index_documents(docs, "test_ns")

                assert docs[0].meta is not None
                assert docs[0].meta.get("namespace") == "test_ns"
                pipeline.close()

    def test_index_from_config(
        self, sample_config, mock_weaviate_db, mock_embedders, sample_documents
    ):
        """Test index_from_config loads documents and indexes them."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.load_documents_from_config",
                    return_value=sample_documents,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)
                mock_weaviate_db.list_tenants.return_value = ["test_ns"]

                result = pipeline.index_from_config("test_ns")

                assert result.success is True
                assert result.data["count"] == 3
                pipeline.close()

    def test_index_documents_with_batch_size(
        self, sample_config, mock_weaviate_db, mock_embedders, sample_documents
    ):
        """Test index_documents uses batch_size from config."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_doc_embedder.run.return_value = {"documents": sample_documents}

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)
                mock_weaviate_db.list_tenants.return_value = ["test_ns"]

                pipeline.index_documents(sample_documents, "test_ns")

                # Check that upsert was called with correct batch_size from config (50)
                mock_weaviate_db.upsert.assert_called_once()
                call_kwargs = mock_weaviate_db.upsert.call_args[1]
                assert call_kwargs.get("batch_size") == 50
                pipeline.close()


class TestWeaviateNamespacePipelineQuery:
    """Test query methods for WeaviateNamespacePipeline."""

    def test_query_namespace(self, sample_config, mock_weaviate_db, mock_embedders):
        """Test query_namespace with timing metrics."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [
            Document(content="Result 1", score=0.9),
            Document(content="Result 2", score=0.8),
        ]
        mock_weaviate_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

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
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test query_namespace calls text embedder."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

                pipeline.query_namespace("test query", "test_ns")

                mock_text_embedder.run.assert_called_once_with(text="test query")
                pipeline.close()

    def test_query_namespace_with_tenant(
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test query_namespace applies tenant parameter."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

                pipeline.query_namespace("test query", "my_namespace", top_k=10)

                # query_namespace calls db.query for the search,
                # then get_namespace_stats uses aggregate (not a query call)
                assert mock_weaviate_db.query.call_count >= 1
                # Check the first call was the search query
                first_call_kwargs = mock_weaviate_db.query.call_args_list[0][1]
                assert first_call_kwargs.get("tenant") == "my_namespace"
                assert first_call_kwargs.get("top_k") == 10
                pipeline.close()

    def test_query_cross_namespace(
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test query_cross_namespace aggregates results from multiple namespaces."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results_ns1 = [Document(content="NS1 Result", score=0.95)]
        mock_results_ns2 = [Document(content="NS2 Result", score=0.85)]

        def mock_query_side_effect(*, vector, top_k, tenant):
            if tenant == "ns1":
                return mock_results_ns1
            if tenant == "ns2":
                return mock_results_ns2
            return []

        mock_weaviate_db.query.side_effect = mock_query_side_effect

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

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
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test query_cross_namespace queries all namespaces when none specified."""
        mock_doc_embedder, mock_text_embedder = mock_embedders

        # Mock list_namespaces to return known values
        mock_weaviate_db.list_tenants.return_value = ["ns1", "ns2"]

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

                # query_cross_namespace calls:
                # 1. list_namespaces() which calls db.list_tenants once
                # 2. For each namespace, query_namespace() which calls:
                #    - db.query for the search
                #    - db.query for get_namespace_stats
                # So for 2 namespaces, we need: 1 + (2 * 2) = 5 calls
                mock_weaviate_db.query.side_effect = [
                    [Document(content="Result 1")],  # ns1 search query
                    [Document(content="doc")],  # ns1 stats query
                    [Document(content="Result 2")],  # ns2 search query
                    [Document(content="doc")],  # ns2 stats query
                ]

                result = pipeline.query_cross_namespace("test query")

                # Should query all listed namespaces
                assert "ns1" in result.namespace_results
                assert "ns2" in result.namespace_results
                pipeline.close()

    def test_query_cross_namespace_timing_comparison(
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test query_cross_namespace includes timing comparison."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [Document(content="Result", score=0.9)]
        mock_weaviate_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

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
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test query_cross_namespace handles empty results."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_weaviate_db.query.return_value = []

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

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
        self, sample_config, mock_weaviate_db, mock_embedders
    ):
        """Test query_cross_namespace with None timing for coverage."""
        mock_doc_embedder, mock_text_embedder = mock_embedders
        mock_results = [Document(content="Result", score=0.9)]
        mock_weaviate_db.query.return_value = mock_results

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            with (
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_text_embedder",
                    return_value=mock_text_embedder,
                ),
                patch(
                    "vectordb.haystack.namespaces.weaviate_namespaces.get_document_embedder",
                    return_value=mock_doc_embedder,
                ),
            ):
                pipeline = WeaviateNamespacePipeline(sample_config)

                # Mock query_namespace to return results with None timing
                # This triggers the else branch in query_cross_namespace
                with patch.object(pipeline, "query_namespace") as mock_query_ns:
                    from vectordb.haystack.namespaces.types import (
                        NamespaceQueryResult,
                    )

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


class TestWeaviateNamespacePipelineRun:
    """Test run method for WeaviateNamespacePipeline."""

    def test_run_method(self, sample_config, mock_weaviate_db):
        """Test run method returns pipeline status."""
        mock_weaviate_db.list_tenants.return_value = ["ns1", "ns2"]

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            result = pipeline.run()

            assert result["success"] is True
            assert "message" in result
            assert "namespaces" in result
            assert "ns1" in result["namespaces"]
            assert "ns2" in result["namespaces"]
            pipeline.close()


class TestWeaviateNamespacePipelineClose:
    """Test close method for WeaviateNamespacePipeline."""

    def test_close_method(self, sample_config, mock_weaviate_db):
        """Test close method clears the database connection."""
        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)
            assert pipeline._db is not None

            pipeline.close()

            assert pipeline._db is None

    def test_close_logs_info(self, sample_config, mock_weaviate_db, caplog):
        """Test close method logs info message."""
        import logging

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db_class:
            mock_db_class.return_value = mock_weaviate_db

            pipeline = WeaviateNamespacePipeline(sample_config)

            with caplog.at_level(logging.INFO):
                pipeline.close()

            assert "Closed Weaviate connection" in caplog.text


class TestWeaviateNamespacePipelineExtended:
    """Extended test suite for additional coverage."""

    def test_context_manager_protocol(self, tmp_path):
        """Test context manager protocol."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
            pipeline:
                name: "test-weaviate"
            weaviate:
                url: "http://localhost:8080"
        """)

        with patch(
            "vectordb.haystack.namespaces.weaviate_namespaces.WeaviateVectorDB"
        ) as mock_db:
            mock_instance = MagicMock()
            mock_db.return_value = mock_instance

            with WeaviateNamespacePipeline(str(config_path)) as pipeline:
                assert pipeline is not None

            # After context exit, _db should be None
            assert pipeline._db is None
