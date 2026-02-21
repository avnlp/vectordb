"""Tests for the base namespace pipeline (LangChain).

This module tests the NamespacePipeline abstract base class that defines the
interface for namespace isolation across vector databases. Namespace isolation
allows logical partitioning of data without requiring separate DB instances.

The NamespacePipeline ABC enforces a consistent contract for:
    - Namespace management (create, delete, list, exists, stats)
    - Document indexing with pre-computed embeddings
    - Namespace-scoped and cross-namespace querying

Test Coverage:
    - ABC inheritance and abstract method enforcement
    - Concrete implementation interface compatibility
    - Namespace isolation behavior
    - Error handling for invalid namespace IDs
"""

from __future__ import annotations

from abc import ABC

import pytest
from langchain_core.documents import Document

from vectordb.langchain.namespaces.base import NamespacePipeline
from vectordb.langchain.namespaces.types import (
    CrossNamespaceResult,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    TenantStatus,
)


class InMemoryNamespacePipeline(NamespacePipeline):
    """Minimal in-memory implementation used to validate the ABC contract."""

    def __init__(self) -> None:
        """Initialize empty namespace storage."""
        self._documents_by_namespace: dict[str, list[Document]] = {}

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        """Validate that namespace ID is non-empty."""
        if not namespace:
            raise ValueError("namespace cannot be empty")

    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create an empty namespace container."""
        self._validate_namespace(namespace)
        self._documents_by_namespace.setdefault(namespace, [])
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="create",
            message="namespace created",
        )

    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete namespace and all stored documents."""
        self._validate_namespace(namespace)
        self._documents_by_namespace.pop(namespace, None)
        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="delete",
            message="namespace deleted",
        )

    def list_namespaces(self) -> list[str]:
        """Return all known namespace identifiers."""
        return list(self._documents_by_namespace.keys())

    def namespace_exists(self, namespace: str) -> bool:
        """Return whether a namespace exists."""
        self._validate_namespace(namespace)
        return namespace in self._documents_by_namespace

    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Return simple namespace statistics from in-memory storage."""
        self._validate_namespace(namespace)
        count = len(self._documents_by_namespace.get(namespace, []))
        return NamespaceStats(
            namespace=namespace,
            document_count=count,
            vector_count=count,
            status=TenantStatus.ACTIVE if count > 0 else TenantStatus.UNKNOWN,
        )

    def index_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        namespace: str,
    ) -> NamespaceOperationResult:
        """Append documents and return indexing result metadata."""
        self._validate_namespace(namespace)
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings count must match")

        if namespace not in self._documents_by_namespace:
            self._documents_by_namespace[namespace] = []
        self._documents_by_namespace[namespace].extend(documents)

        return NamespaceOperationResult(
            success=True,
            namespace=namespace,
            operation="index",
            message="documents indexed",
            data={"count": len(documents)},
        )

    def query_namespace(
        self,
        query: str,
        namespace: str,
        top_k: int = 10,
    ) -> list[NamespaceQueryResult]:
        """Run simple substring search inside a single namespace."""
        self._validate_namespace(namespace)
        if namespace not in self._documents_by_namespace:
            return []

        query_lower = query.lower()
        docs = self._documents_by_namespace[namespace]
        matched = [doc for doc in docs if query_lower in doc.page_content.lower()]
        return [
            NamespaceQueryResult(
                document=doc,
                relevance_score=1.0,
                rank=idx + 1,
                namespace=namespace,
            )
            for idx, doc in enumerate(matched[:top_k])
        ]

    def query_cross_namespace(
        self,
        query: str,
        namespaces: list[str] | None = None,
        top_k: int = 10,
    ) -> CrossNamespaceResult:
        """Run namespace queries and aggregate into a cross-namespace result."""
        target_namespaces = (
            namespaces if namespaces is not None else self.list_namespaces()
        )
        namespace_results = {
            namespace: self.query_namespace(query, namespace, top_k)
            for namespace in target_namespaces
        }
        return CrossNamespaceResult(
            query=query,
            namespace_results=namespace_results,
            timing_comparison=[],
            total_time_ms=0.0,
        )


class TestNamespacePipelineABC:
    """Tests for NamespacePipeline abstract base class compliance."""

    def test_is_abstract_base_class(self):
        """Validate NamespacePipeline inherits from ABC."""
        assert issubclass(NamespacePipeline, ABC)

    def test_cannot_instantiate_directly(self):
        """Validate NamespacePipeline cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NamespacePipeline()

    def test_has_required_abstract_methods(self):
        """Validate all required abstract methods are declared."""
        abstract_methods = getattr(NamespacePipeline, "__abstractmethods__", set())
        required_methods = {
            "create_namespace",
            "delete_namespace",
            "list_namespaces",
            "namespace_exists",
            "get_namespace_stats",
            "index_documents",
            "query_namespace",
            "query_cross_namespace",
        }
        assert required_methods.issubset(abstract_methods)


class TestConcreteNamespacePipelineImplementation:
    """Tests concrete in-memory implementation against the ABC interface."""

    def test_concrete_implementation_can_be_created(self):
        """Validate a concrete NamespacePipeline implementation instantiates."""
        pipeline = InMemoryNamespacePipeline()
        assert isinstance(pipeline, NamespacePipeline)

    def test_concrete_implementation_methods_work(self):
        """Validate the concrete implementation returns expected method types."""
        pipeline = InMemoryNamespacePipeline()

        create_result = pipeline.create_namespace("ns_a")
        assert isinstance(create_result, NamespaceOperationResult)
        assert create_result.success is True

        docs = [Document(page_content="python namespace", metadata={"id": "1"})]
        embeddings = [[0.1] * 3]
        index_result = pipeline.index_documents(docs, embeddings, "ns_a")
        assert isinstance(index_result, NamespaceOperationResult)
        assert index_result.data == {"count": 1}

        namespaces = pipeline.list_namespaces()
        assert isinstance(namespaces, list)
        assert namespaces == ["ns_a"]

        query_results = pipeline.query_namespace("python", "ns_a")
        assert isinstance(query_results, list)
        assert len(query_results) == 1
        assert isinstance(query_results[0], NamespaceQueryResult)


class TestNamespaceIsolation:
    """Tests namespace-level data isolation behavior."""

    def test_namespace_isolation_in_concrete_implementation(self):
        """Validate documents are isolated across namespaces."""
        pipeline = InMemoryNamespacePipeline()

        pipeline.index_documents(
            [Document(page_content="alpha content", metadata={})],
            [[0.1] * 3],
            "ns_alpha",
        )
        pipeline.index_documents(
            [Document(page_content="beta content", metadata={})],
            [[0.2] * 3],
            "ns_beta",
        )

        alpha_results = pipeline.query_namespace("alpha", "ns_alpha")
        beta_results = pipeline.query_namespace("alpha", "ns_beta")

        assert len(alpha_results) == 1
        assert alpha_results[0].document.page_content == "alpha content"
        assert beta_results == []

    def test_delete_namespace_does_not_affect_other_namespaces(self):
        """Validate deleting one namespace does not affect others."""
        pipeline = InMemoryNamespacePipeline()

        pipeline.index_documents(
            [Document(page_content="namespace a doc", metadata={})],
            [[0.1] * 3],
            "ns_a",
        )
        pipeline.index_documents(
            [Document(page_content="namespace b doc", metadata={})],
            [[0.2] * 3],
            "ns_b",
        )

        pipeline.delete_namespace("ns_a")

        assert pipeline.namespace_exists("ns_b") is True
        assert pipeline.query_namespace("namespace", "ns_b")
        assert pipeline.query_namespace("namespace", "ns_a") == []


class TestNamespacePipelineErrorHandling:
    """Tests validation and edge-case behavior for namespace operations."""

    def test_empty_namespace_id_handling(self):
        """Validate empty namespace IDs raise ValueError."""
        pipeline = InMemoryNamespacePipeline()

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            pipeline.create_namespace("")

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            pipeline.delete_namespace("")

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            pipeline.namespace_exists("")

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            pipeline.get_namespace_stats("")

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            pipeline.index_documents([], [], "")

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            pipeline.query_namespace("anything", "")

    def test_query_nonexistent_namespace_returns_empty(self):
        """Validate querying a missing namespace returns an empty list."""
        pipeline = InMemoryNamespacePipeline()
        results = pipeline.query_namespace("missing", "not_created")
        assert results == []
