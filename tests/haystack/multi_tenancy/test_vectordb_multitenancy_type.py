"""Tests for multi-tenancy type utilities."""

from datetime import datetime

import pytest

from vectordb.haystack.multi_tenancy.vectordb_multitenancy_type import (
    MultitenancyError,
    MultitenancyTimingMetrics,
    TenantConnectionError,
    TenantExistsError,
    TenantIndexResult,
    TenantIsolationConfig,
    TenantIsolationStrategy,
    TenantNotFoundError,
    TenantOperationNotSupportedError,
    TenantOperationResult,
    TenantQueryResult,
    TenantRAGResult,
    TenantRetrievalResult,
    TenantStats,
    TenantStatus,
)


class TestTenantIsolationStrategy:
    """Tests for TenantIsolationStrategy enum."""

    def test_all_strategies_defined(self):
        """Test that all expected strategies are defined."""
        assert TenantIsolationStrategy.PARTITION_KEY.value == "partition_key"
        assert (
            TenantIsolationStrategy.NATIVE_MULTITENANCY.value == "native_multitenancy"
        )
        assert TenantIsolationStrategy.NAMESPACE.value == "namespace"
        assert TenantIsolationStrategy.TIERED.value == "tiered"
        assert TenantIsolationStrategy.DATABASE_SCOPING.value == "database_scoping"

    def test_strategy_count(self):
        """Test that exactly 5 strategies are defined."""
        assert len(TenantIsolationStrategy) == 5


class TestTenantStatus:
    """Tests for TenantStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all expected statuses are defined."""
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.INACTIVE.value == "inactive"
        assert TenantStatus.OFFLOADED.value == "offloaded"
        assert TenantStatus.UNKNOWN.value == "unknown"

    def test_status_count(self):
        """Test that exactly 4 statuses are defined."""
        assert len(TenantStatus) == 4


class TestTenantIsolationConfig:
    """Tests for TenantIsolationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TenantIsolationConfig(strategy="partition_key")
        assert config.strategy == "partition_key"
        assert config.field_name == "tenant_id"
        assert config.auto_create_tenant is True
        assert config.partition_key_isolation is False
        assert config.num_partitions == 64

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TenantIsolationConfig(
            strategy="namespace",
            field_name="org_id",
            auto_create_tenant=False,
            partition_key_isolation=True,
            num_partitions=128,
        )
        assert config.strategy == "namespace"
        assert config.field_name == "org_id"
        assert config.auto_create_tenant is False
        assert config.partition_key_isolation is True
        assert config.num_partitions == 128

    def test_strategy_literal_types(self):
        """Test that strategy accepts only valid literal types."""
        for strategy in [
            "partition_key",
            "namespace",
            "native_multitenancy",
            "tiered",
            "database_scoping",
        ]:
            config = TenantIsolationConfig(strategy=strategy)
            assert config.strategy == strategy


class TestMultitenancyTimingMetrics:
    """Tests for MultitenancyTimingMetrics dataclass."""

    def test_create_timing_metrics(self):
        """Test creating timing metrics."""
        metrics = MultitenancyTimingMetrics(
            tenant_resolution_ms=10.5,
            index_operation_ms=100.0,
            retrieval_ms=50.0,
            total_ms=160.5,
            tenant_id="tenant_123",
            num_documents=10,
        )
        assert metrics.tenant_resolution_ms == 10.5
        assert metrics.index_operation_ms == 100.0
        assert metrics.retrieval_ms == 50.0
        assert metrics.total_ms == 160.5
        assert metrics.tenant_id == "tenant_123"
        assert metrics.num_documents == 10

    def test_timing_metrics_with_zero_values(self):
        """Test timing metrics with zero values."""
        metrics = MultitenancyTimingMetrics(
            tenant_resolution_ms=0.0,
            index_operation_ms=0.0,
            retrieval_ms=0.0,
            total_ms=0.0,
            tenant_id="tenant_456",
            num_documents=0,
        )
        assert metrics.tenant_resolution_ms == 0.0
        assert metrics.num_documents == 0


class TestTenantIndexResult:
    """Tests for TenantIndexResult dataclass."""

    def test_create_index_result(self):
        """Test creating index result."""
        timing = MultitenancyTimingMetrics(
            tenant_resolution_ms=5.0,
            index_operation_ms=200.0,
            retrieval_ms=0.0,
            total_ms=205.0,
            tenant_id="tenant_123",
            num_documents=50,
        )
        result = TenantIndexResult(
            tenant_id="tenant_123",
            documents_indexed=50,
            collection_name="test_collection",
            timing=timing,
            success=True,
            message="Successfully indexed 50 documents",
        )
        assert result.tenant_id == "tenant_123"
        assert result.documents_indexed == 50
        assert result.collection_name == "test_collection"
        assert result.success is True
        assert result.message == "Successfully indexed 50 documents"

    def test_index_result_default_values(self):
        """Test index result default values."""
        timing = MultitenancyTimingMetrics(
            tenant_resolution_ms=0.0,
            index_operation_ms=0.0,
            retrieval_ms=0.0,
            total_ms=0.0,
            tenant_id="tenant_123",
            num_documents=0,
        )
        result = TenantIndexResult(
            tenant_id="tenant_123",
            documents_indexed=0,
            collection_name="test",
            timing=timing,
        )
        assert result.success is True
        assert result.message == ""


class TestTenantRetrievalResult:
    """Tests for TenantRetrievalResult dataclass."""

    def test_create_retrieval_result(self):
        """Test creating retrieval result with mock documents."""
        from haystack import Document

        mock_docs = [
            Document(content="Test document 1", id="1"),
            Document(content="Test document 2", id="2"),
        ]
        timing = MultitenancyTimingMetrics(
            tenant_resolution_ms=5.0,
            index_operation_ms=0.0,
            retrieval_ms=100.0,
            total_ms=105.0,
            tenant_id="tenant_123",
            num_documents=2,
        )
        result = TenantRetrievalResult(
            tenant_id="tenant_123",
            query="test query",
            documents=mock_docs,
            scores=[0.9, 0.8],
            timing=timing,
        )
        assert result.tenant_id == "tenant_123"
        assert result.query == "test query"
        assert len(result.documents) == 2
        assert result.scores == [0.9, 0.8]


class TestTenantRAGResult:
    """Tests for TenantRAGResult dataclass."""

    def test_create_rag_result(self):
        """Test creating RAG result."""
        from haystack import Document

        mock_docs = [Document(content="Context document", id="1")]
        timing = MultitenancyTimingMetrics(
            tenant_resolution_ms=5.0,
            index_operation_ms=0.0,
            retrieval_ms=100.0,
            total_ms=500.0,
            tenant_id="tenant_123",
            num_documents=1,
        )
        result = TenantRAGResult(
            tenant_id="tenant_123",
            query="What is the answer?",
            retrieved_documents=mock_docs,
            generated_response="The answer is 42.",
            timing=timing,
            retrieval_scores=[0.95],
        )
        assert result.tenant_id == "tenant_123"
        assert result.generated_response == "The answer is 42."
        assert result.retrieval_scores == [0.95]

    def test_rag_result_default_scores(self):
        """Test RAG result default retrieval scores."""
        timing = MultitenancyTimingMetrics(
            tenant_resolution_ms=0.0,
            index_operation_ms=0.0,
            retrieval_ms=0.0,
            total_ms=0.0,
            tenant_id="tenant_123",
            num_documents=0,
        )
        result = TenantRAGResult(
            tenant_id="tenant_123",
            query="test",
            retrieved_documents=[],
            generated_response="response",
            timing=timing,
        )
        assert result.retrieval_scores == []


class TestTenantQueryResult:
    """Tests for TenantQueryResult dataclass."""

    def test_create_query_result(self):
        """Test creating query result."""
        from haystack import Document

        doc = Document(content="Retrieved document", id="doc1")
        result = TenantQueryResult(
            document=doc,
            relevance_score=0.95,
            rank=1,
            tenant_id="tenant_123",
        )
        assert result.document == doc
        assert result.relevance_score == 0.95
        assert result.rank == 1
        assert result.tenant_id == "tenant_123"


class TestTenantOperationResult:
    """Tests for TenantOperationResult dataclass."""

    def test_create_operation_result(self):
        """Test creating operation result."""
        result = TenantOperationResult(
            success=True,
            tenant_id="tenant_123",
            operation="create",
            message="Tenant created successfully",
            data={"document_count": 100},
        )
        assert result.success is True
        assert result.tenant_id == "tenant_123"
        assert result.operation == "create"
        assert result.data == {"document_count": 100}

    def test_operation_result_default_values(self):
        """Test operation result default values."""
        result = TenantOperationResult(
            success=False,
            tenant_id="tenant_123",
            operation="delete",
        )
        assert result.message == ""
        assert result.data is None


class TestTenantStats:
    """Tests for TenantStats dataclass."""

    def test_create_tenant_stats(self):
        """Test creating tenant stats."""
        now = datetime.now()
        stats = TenantStats(
            tenant_id="tenant_123",
            document_count=500,
            vector_count=500,
            status=TenantStatus.ACTIVE,
            created_at=now,
            last_updated=now,
            size_bytes=1024000,
        )
        assert stats.tenant_id == "tenant_123"
        assert stats.document_count == 500
        assert stats.vector_count == 500
        assert stats.status == TenantStatus.ACTIVE
        assert stats.created_at == now
        assert stats.size_bytes == 1024000

    def test_tenant_stats_default_values(self):
        """Test tenant stats default values."""
        stats = TenantStats(
            tenant_id="tenant_123",
            document_count=100,
        )
        assert stats.vector_count == 0
        assert stats.status == TenantStatus.ACTIVE
        assert stats.created_at is None
        assert stats.last_updated is None
        assert stats.size_bytes == 0


class TestMultitenancyExceptions:
    """Tests for multi-tenancy exception classes."""

    def test_multitenancy_error(self):
        """Test base MultitenancyError exception."""
        with pytest.raises(MultitenancyError):
            raise MultitenancyError("Base error")

    def test_tenant_not_found_error(self):
        """Test TenantNotFoundError exception."""
        with pytest.raises(TenantNotFoundError):
            raise TenantNotFoundError("Tenant not found")

    def test_tenant_exists_error(self):
        """Test TenantExistsError exception."""
        with pytest.raises(TenantExistsError):
            raise TenantExistsError("Tenant already exists")

    def test_tenant_operation_not_supported_error(self):
        """Test TenantOperationNotSupportedError exception."""
        with pytest.raises(TenantOperationNotSupportedError):
            raise TenantOperationNotSupportedError("Operation not supported")

    def test_tenant_connection_error(self):
        """Test TenantConnectionError exception."""
        with pytest.raises(TenantConnectionError):
            raise TenantConnectionError("Connection failed")

    def test_exception_inheritance(self):
        """Test that all exceptions inherit from MultitenancyError."""
        assert issubclass(TenantNotFoundError, MultitenancyError)
        assert issubclass(TenantExistsError, MultitenancyError)
        assert issubclass(TenantOperationNotSupportedError, MultitenancyError)
        assert issubclass(TenantConnectionError, MultitenancyError)
