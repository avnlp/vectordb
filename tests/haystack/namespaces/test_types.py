"""Tests for namespace types and utilities.

Tests cover:
- Enums: IsolationStrategy, TenantStatus
- Dataclasses: NamespaceConfig, NamespaceStats, NamespaceTimingMetrics,
  NamespaceQueryResult, CrossNamespaceComparison, CrossNamespaceResult,
  NamespaceOperationResult
- Exceptions: NamespaceError and subclasses
- NamespaceNameGenerator utilities
- QuerySampler functionality
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.namespaces.types import (
    CrossNamespaceComparison,
    CrossNamespaceResult,
    IsolationStrategy,
    NamespaceConfig,
    NamespaceConnectionError,
    NamespaceError,
    NamespaceExistsError,
    NamespaceNameGenerator,
    NamespaceNotFoundError,
    NamespaceOperationNotSupportedError,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    NamespaceTimingMetrics,
    QuerySampler,
    TenantStatus,
)


class TestQuerySampler:
    """Tests for QuerySampler class."""

    @pytest.fixture
    def sample_documents_with_questions(self) -> list[Document]:
        """Create documents with question metadata."""
        return [
            Document(content="Content A", meta={"question": "What is ML?"}),
            Document(content="Content B", meta={"question": "How does AI work?"}),
            Document(content="Content C", meta={"question": "What is deep learning?"}),
            Document(content="Content D", meta={"question": "Explain NLP."}),
            Document(content="Content E", meta={"question": "Define LLMs."}),
        ]

    @pytest.fixture
    def sample_documents_without_questions(self) -> list[Document]:
        """Create documents without question metadata."""
        return [
            Document(
                content="This is content A that is longer than 100 characters " * 3
            ),
            Document(
                content="This is content B that is longer than 100 characters " * 3
            ),
            Document(content="Short C"),
        ]

    @pytest.fixture
    def mixed_documents(self) -> list[Document]:
        """Create documents with mixed metadata."""
        return [
            Document(content="Has question", meta={"question": "Query from meta?"}),
            Document(content="No question here, fallback to content slicing"),
            Document(
                content="Another with question", meta={"question": "Another query?"}
            ),
        ]

    def test_sample_from_documents_with_question_field(
        self, sample_documents_with_questions: list[Document]
    ) -> None:
        """Test sampling queries from question metadata field."""
        result = QuerySampler.sample_from_documents(
            sample_documents_with_questions, sample_size=3, seed=42
        )

        assert len(result) == 3
        assert all(isinstance(q, str) for q in result)
        # Verify we got actual question values from metadata
        expected_queries = {
            "What is ML?",
            "How does AI work?",
            "What is deep learning?",
            "Explain NLP.",
            "Define LLMs.",
        }
        assert all(q in expected_queries for q in result)

    def test_sample_from_documents_uses_content_fallback(
        self, sample_documents_without_questions: list[Document]
    ) -> None:
        """Test fallback to content slicing when question field not in meta."""
        result = QuerySampler.sample_from_documents(
            sample_documents_without_questions, sample_size=2, seed=42
        )

        assert len(result) == 2
        assert all(isinstance(q, str) for q in result)
        assert all(len(q) <= 100 for q in result)

    def test_sample_from_documents_empty_documents_returns_empty(self) -> None:
        """Test empty documents list returns empty queries."""
        result = QuerySampler.sample_from_documents([], sample_size=5)

        assert result == []

    def test_sample_from_documents_no_valid_queries_returns_empty(self) -> None:
        """Test documents with no meta and no content returns empty."""
        docs = [Document(content="", meta={})]

        result = QuerySampler.sample_from_documents(docs, sample_size=5)

        assert result == []

    def test_sample_from_documents_sample_size_adjustment(
        self, sample_documents_with_questions: list[Document]
    ) -> None:
        """Test sample size adjusted when greater than available queries."""
        result = QuerySampler.sample_from_documents(
            sample_documents_with_questions, sample_size=10, seed=42
        )

        assert len(result) == 5  # Only 5 documents available

    def test_sample_from_documents_deterministic_with_seed(
        self, sample_documents_with_questions: list[Document]
    ) -> None:
        """Test sampling is deterministic with same seed."""
        result1 = QuerySampler.sample_from_documents(
            sample_documents_with_questions, sample_size=3, seed=123
        )
        result2 = QuerySampler.sample_from_documents(
            sample_documents_with_questions, sample_size=3, seed=123
        )

        assert result1 == result2

    def test_sample_from_documents_custom_query_field(self) -> None:
        """Test sampling with custom query field."""
        docs = [
            Document(content="C", meta={"query": "Custom query 1"}),
            Document(content="D", meta={"query": "Custom query 2"}),
        ]

        result = QuerySampler.sample_from_documents(
            docs, sample_size=2, query_field="query", seed=42
        )

        assert len(result) == 2
        assert "Custom query 1" in result
        assert "Custom query 2" in result

    def test_sample_from_documents_mixed_sources(
        self, mixed_documents: list[Document]
    ) -> None:
        """Test sampling includes both meta and content fallback sources."""
        result = QuerySampler.sample_from_documents(
            mixed_documents, sample_size=3, seed=42
        )

        assert len(result) == 3
        # Should have queries from both sources
        meta_queries = [q for q in result if "query" in q.lower()]
        content_queries = [q for q in result if "fallback" in q.lower()]

        assert len(meta_queries) >= 1
        assert len(content_queries) >= 1

    @patch("random.sample")
    def test_sample_from_documents_calls_random_sample(
        self, mock_sample: MagicMock, sample_documents_with_questions: list[Document]
    ) -> None:
        """Test that random.sample is called with correct arguments."""
        mock_sample.return_value = ["Query 1", "Query 2"]

        QuerySampler.sample_from_documents(
            sample_documents_with_questions, sample_size=2, seed=42
        )

        mock_sample.assert_called_once()
        call_args = mock_sample.call_args
        assert call_args[0][1] == 2  # sample_size


class TestNamespaceNameGenerator:
    """Tests for NamespaceNameGenerator class."""

    def test_from_split_basic(self) -> None:
        """Test generating name from dataset and split."""
        result = NamespaceNameGenerator.from_split("arc", "train")

        assert result == "arc_train"

    def test_from_split_with_prefix(self) -> None:
        """Test generating name with prefix."""
        result = NamespaceNameGenerator.from_split("arc", "train", "qa")

        assert result == "qa_arc_train"

    def test_from_split_empty_prefix(self) -> None:
        """Test generating name with empty prefix."""
        result = NamespaceNameGenerator.from_split("arc", "train", "")

        assert result == "arc_train"

    def test_from_ticker_basic(self) -> None:
        """Test generating name from ticker."""
        result = NamespaceNameGenerator.from_ticker("AAPL")

        assert result == "earnings_AAPL"

    def test_from_ticker_custom_prefix(self) -> None:
        """Test generating name with custom prefix."""
        result = NamespaceNameGenerator.from_ticker("GOOGL", "stocks")

        assert result == "stocks_GOOGL"

    def test_from_config_extracts_names(self) -> None:
        """Test extracting namespace names from config."""
        config = {
            "namespaces": {
                "definitions": [
                    {"name": "ns1", "description": "First"},
                    {"name": "ns2", "description": "Second"},
                ]
            }
        }

        result = NamespaceNameGenerator.from_config(config)

        assert result == ["ns1", "ns2"]

    def test_from_config_empty_definitions(self) -> None:
        """Test empty definitions returns empty list."""
        config = {"namespaces": {"definitions": []}}

        result = NamespaceNameGenerator.from_config(config)

        assert result == []

    def test_from_config_missing_namespaces_key(self) -> None:
        """Test missing namespaces key returns empty list."""
        config = {"other": "data"}

        result = NamespaceNameGenerator.from_config(config)

        assert result == []

    def test_from_config_missing_definitions(self) -> None:
        """Test missing definitions key returns empty list."""
        config = {"namespaces": {}}

        result = NamespaceNameGenerator.from_config(config)

        assert result == []

    def test_from_config_skips_missing_name(self) -> None:
        """Test entries without name are skipped."""
        config = {
            "namespaces": {
                "definitions": [
                    {"name": "ns1"},
                    {"description": "No name"},
                    {"name": "ns2"},
                ]
            }
        }

        result = NamespaceNameGenerator.from_config(config)

        assert result == ["ns1", "ns2"]

    def test_from_config_complex_structure(self) -> None:
        """Test extracting from complex nested config."""
        config = {
            "database": {"host": "localhost"},
            "namespaces": {
                "definitions": [
                    {"name": "train_ns", "split": "train"},
                    {"name": "test_ns", "split": "test"},
                    {"name": "val_ns", "split": "validation"},
                ],
                "settings": {"isolation": "tenant"},
            },
        }

        result = NamespaceNameGenerator.from_config(config)

        assert len(result) == 3
        assert "train_ns" in result
        assert "test_ns" in result
        assert "val_ns" in result


class TestNamespaceDataclasses:
    """Tests for namespace dataclasses."""

    def test_namespace_config_defaults(self) -> None:
        """Test NamespaceConfig with default values."""
        config = NamespaceConfig(name="test_ns")

        assert config.name == "test_ns"
        assert config.description == ""
        assert config.split == ""
        assert config.metadata == {}

    def test_namespace_config_full(self) -> None:
        """Test NamespaceConfig with all values."""
        config = NamespaceConfig(
            name="test_ns",
            description="Test namespace",
            split="train",
            metadata={"key": "value"},
        )

        assert config.name == "test_ns"
        assert config.description == "Test namespace"
        assert config.split == "train"
        assert config.metadata == {"key": "value"}

    def test_namespace_stats_defaults(self) -> None:
        """Test NamespaceStats with default values."""
        stats = NamespaceStats(namespace="test_ns", document_count=100)

        assert stats.namespace == "test_ns"
        assert stats.document_count == 100
        assert stats.vector_count == 0
        assert stats.status == TenantStatus.ACTIVE
        assert stats.created_at is None
        assert stats.last_updated is None
        assert stats.size_bytes == 0

    def test_namespace_stats_full(self) -> None:
        """Test NamespaceStats with all values."""
        now = datetime.now()
        stats = NamespaceStats(
            namespace="test_ns",
            document_count=100,
            vector_count=100,
            status=TenantStatus.INACTIVE,
            created_at=now,
            last_updated=now,
            size_bytes=1024,
        )

        assert stats.vector_count == 100
        assert stats.status == TenantStatus.INACTIVE
        assert stats.created_at == now
        assert stats.last_updated == now
        assert stats.size_bytes == 1024

    def test_namespace_timing_metrics(self) -> None:
        """Test NamespaceTimingMetrics dataclass."""
        metrics = NamespaceTimingMetrics(
            namespace_lookup_ms=10.5,
            vector_search_ms=25.3,
            total_ms=35.8,
            documents_searched=1000,
            documents_returned=10,
        )

        assert metrics.namespace_lookup_ms == 10.5
        assert metrics.vector_search_ms == 25.3
        assert metrics.total_ms == 35.8
        assert metrics.documents_searched == 1000
        assert metrics.documents_returned == 10

    def test_namespace_query_result_defaults(self) -> None:
        """Test NamespaceQueryResult with default timing."""
        doc = Document(content="Test")
        result = NamespaceQueryResult(
            document=doc,
            relevance_score=0.95,
            rank=1,
            namespace="test_ns",
        )

        assert result.document == doc
        assert result.relevance_score == 0.95
        assert result.rank == 1
        assert result.namespace == "test_ns"
        assert result.timing is None

    def test_namespace_query_result_with_timing(self) -> None:
        """Test NamespaceQueryResult with timing metrics."""
        doc = Document(content="Test")
        timing = NamespaceTimingMetrics(
            namespace_lookup_ms=5.0,
            vector_search_ms=10.0,
            total_ms=15.0,
            documents_searched=100,
            documents_returned=5,
        )
        result = NamespaceQueryResult(
            document=doc,
            relevance_score=0.95,
            rank=1,
            namespace="test_ns",
            timing=timing,
        )

        assert result.timing == timing

    def test_cross_namespace_comparison(self) -> None:
        """Test CrossNamespaceComparison dataclass."""
        timing = NamespaceTimingMetrics(
            namespace_lookup_ms=5.0,
            vector_search_ms=10.0,
            total_ms=15.0,
            documents_searched=100,
            documents_returned=5,
        )
        comparison = CrossNamespaceComparison(
            namespace="test_ns",
            timing=timing,
            result_count=5,
            top_score=0.98,
        )

        assert comparison.namespace == "test_ns"
        assert comparison.timing == timing
        assert comparison.result_count == 5
        assert comparison.top_score == 0.98

    def test_cross_namespace_result(self) -> None:
        """Test CrossNamespaceResult dataclass."""
        result = CrossNamespaceResult(
            query="test query",
            namespace_results={"ns1": [], "ns2": []},
            timing_comparison=[],
            total_time_ms=100.0,
        )

        assert result.query == "test query"
        assert result.namespace_results == {"ns1": [], "ns2": []}
        assert result.timing_comparison == []
        assert result.total_time_ms == 100.0

    def test_namespace_operation_result_defaults(self) -> None:
        """Test NamespaceOperationResult with default message and data."""
        result = NamespaceOperationResult(
            success=True,
            namespace="test_ns",
            operation="create",
        )

        assert result.success is True
        assert result.namespace == "test_ns"
        assert result.operation == "create"
        assert result.message == ""
        assert result.data is None

    def test_namespace_operation_result_full(self) -> None:
        """Test NamespaceOperationResult with all fields."""
        result = NamespaceOperationResult(
            success=False,
            namespace="test_ns",
            operation="delete",
            message="Namespace not found",
            data={"error_code": 404},
        )

        assert result.success is False
        assert result.operation == "delete"
        assert result.message == "Namespace not found"
        assert result.data == {"error_code": 404}


class TestNamespaceExceptions:
    """Tests for namespace exceptions."""

    def test_namespace_error_is_exception(self) -> None:
        """Test NamespaceError is an Exception subclass."""
        assert issubclass(NamespaceError, Exception)

    def test_namespace_not_found_error_is_namespace_error(self) -> None:
        """Test NamespaceNotFoundError is a NamespaceError subclass."""
        assert issubclass(NamespaceNotFoundError, NamespaceError)

    def test_namespace_exists_error_is_namespace_error(self) -> None:
        """Test NamespaceExistsError is a NamespaceError subclass."""
        assert issubclass(NamespaceExistsError, NamespaceError)

    def test_namespace_operation_not_supported_error_is_namespace_error(self) -> None:
        """Test NamespaceOperationNotSupportedError is a NamespaceError subclass."""
        assert issubclass(NamespaceOperationNotSupportedError, NamespaceError)

    def test_namespace_connection_error_is_namespace_error(self) -> None:
        """Test NamespaceConnectionError is a NamespaceError subclass."""
        assert issubclass(NamespaceConnectionError, NamespaceError)

    def test_namespace_not_found_error_raised(self) -> None:
        """Test NamespaceNotFoundError can be raised and caught."""
        with pytest.raises(NamespaceNotFoundError):
            raise NamespaceNotFoundError("Namespace 'foo' not found")

    def test_namespace_exists_error_raised(self) -> None:
        """Test NamespaceExistsError can be raised and caught."""
        with pytest.raises(NamespaceExistsError):
            raise NamespaceExistsError("Namespace 'foo' already exists")

    def test_namespace_operation_not_supported_error_raised(self) -> None:
        """Test NamespaceOperationNotSupportedError can be raised and caught."""
        with pytest.raises(NamespaceOperationNotSupportedError):
            raise NamespaceOperationNotSupportedError("Operation not supported")

    def test_namespace_connection_error_raised(self) -> None:
        """Test NamespaceConnectionError can be raised and caught."""
        with pytest.raises(NamespaceConnectionError):
            raise NamespaceConnectionError("Failed to connect")

    def test_namespace_error_caught_as_base(self) -> None:
        """Test all subclasses can be caught as NamespaceError."""
        exceptions = [
            NamespaceNotFoundError("not found"),
            NamespaceExistsError("exists"),
            NamespaceOperationNotSupportedError("not supported"),
            NamespaceConnectionError("connection failed"),
        ]

        for exc in exceptions:
            with pytest.raises(NamespaceError):
                raise exc


class TestIsolationStrategy:
    """Tests for IsolationStrategy enum."""

    def test_isolation_strategy_values(self) -> None:
        """Test IsolationStrategy enum values."""
        assert IsolationStrategy.NAMESPACE.value == "namespace"
        assert IsolationStrategy.TENANT.value == "tenant"
        assert IsolationStrategy.PARTITION_KEY.value == "partition_key"
        assert IsolationStrategy.PAYLOAD_FILTER.value == "payload_filter"
        assert IsolationStrategy.COLLECTION.value == "collection"

    def test_isolation_strategy_comparison(self) -> None:
        """Test IsolationStrategy enum comparison."""
        assert IsolationStrategy.NAMESPACE == IsolationStrategy.NAMESPACE
        assert IsolationStrategy.TENANT != IsolationStrategy.NAMESPACE


class TestTenantStatus:
    """Tests for TenantStatus enum."""

    def test_tenant_status_values(self) -> None:
        """Test TenantStatus enum values."""
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.INACTIVE.value == "inactive"
        assert TenantStatus.OFFLOADED.value == "offloaded"
        assert TenantStatus.UNKNOWN.value == "unknown"

    def test_tenant_status_comparison(self) -> None:
        """Test TenantStatus enum comparison."""
        assert TenantStatus.ACTIVE == TenantStatus.ACTIVE
        assert TenantStatus.INACTIVE != TenantStatus.ACTIVE
