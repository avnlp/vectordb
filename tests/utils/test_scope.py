"""Tests for scope isolation utilities.

This module tests utilities for multi-tenant data isolation in vector databases.
Scope isolation enables partitioning data by tenant, namespace, or other
organizational boundaries without requiring separate collections.

Tested functions:
    inject_scope_to_metadata: Add scope identifiers to document metadata.
    inject_scope_to_filter: Add scope constraints to query filters.
    build_scope_filter_expr: Build Milvus-style filter expressions with scope.

Test coverage includes:
    - Injection into empty and existing metadata/filters
    - Custom scope field names
    - Handling of None scope values (bypass isolation)
    - Complex filter expression combinations
"""

from vectordb.utils.scope import (
    build_scope_filter_expr,
    inject_scope_to_filter,
    inject_scope_to_metadata,
)


class TestInjectScopeToMetadata:
    """Test suite for inject_scope_to_metadata function.

    Tests cover scope injection into document metadata for multi-tenant
    data storage.
    """

    def test_inject_scope_to_empty_metadata(self) -> None:
        """Test injecting scope into empty metadata."""
        result = inject_scope_to_metadata(None, "tenant_a")
        assert result == {"tenant_id": "tenant_a"}

    def test_inject_scope_to_existing_metadata(self) -> None:
        """Test injecting scope into existing metadata."""
        metadata = {"source": "api", "version": 1}
        result = inject_scope_to_metadata(metadata, "tenant_b")
        assert result["tenant_id"] == "tenant_b"
        assert result["source"] == "api"
        assert result["version"] == 1

    def test_inject_scope_with_custom_field(self) -> None:
        """Test injecting scope with custom field name."""
        result = inject_scope_to_metadata(None, "namespace_x", scope_field="namespace")
        assert result == {"namespace": "namespace_x"}

    def test_no_scope_returns_original_metadata(self) -> None:
        """Test that None scope returns original metadata."""
        metadata = {"source": "api"}
        result = inject_scope_to_metadata(metadata, None)
        assert result == metadata

    def test_no_scope_on_empty_metadata(self) -> None:
        """Test that None scope on empty metadata returns empty dict."""
        result = inject_scope_to_metadata(None, None)
        assert result == {}


class TestInjectScopeToFilter:
    """Test suite for inject_scope_to_filter function.

    Tests cover scope constraint injection into query filters for
    tenant-isolated retrieval.
    """

    def test_inject_scope_to_empty_filter(self) -> None:
        """Test injecting scope into empty filter."""
        result = inject_scope_to_filter(None, "tenant_a")
        assert result == {"tenant_id": {"$eq": "tenant_a"}}

    def test_inject_scope_to_existing_filter(self) -> None:
        """Test injecting scope into existing filter."""
        filters = {"category": "news"}
        result = inject_scope_to_filter(filters, "tenant_a")
        assert result == {
            "$and": [
                {"tenant_id": {"$eq": "tenant_a"}},
                {"category": "news"},
            ]
        }

    def test_inject_scope_with_custom_field(self) -> None:
        """Test injecting scope with custom field name."""
        result = inject_scope_to_filter(None, "namespace_x", scope_field="namespace")
        assert result == {"namespace": {"$eq": "namespace_x"}}

    def test_no_scope_returns_original_filter(self) -> None:
        """Test that None scope returns original filter."""
        filters = {"category": "news"}
        result = inject_scope_to_filter(filters, None)
        assert result == filters

    def test_no_scope_on_empty_filter(self) -> None:
        """Test that None scope on empty filter returns None."""
        result = inject_scope_to_filter(None, None)
        assert result is None


class TestBuildScopeFilterExpr:
    """Test suite for build_scope_filter_expr function.

    Tests cover Milvus-style filter expression building with scope
    constraints for databases using expression-based filtering.
    """

    def test_build_scope_expr_only(self) -> None:
        """Test building scope expression only."""
        result = build_scope_filter_expr("tenant_a")
        assert result == 'namespace == "tenant_a"'

    def test_build_scope_with_existing_expr(self) -> None:
        """Test building scope expression combined with existing."""
        result = build_scope_filter_expr("tenant_a", existing_expr='category == "news"')
        assert result == '(category == "news") and namespace == "tenant_a"'

    def test_build_scope_with_custom_field(self) -> None:
        """Test building scope expression with custom field."""
        result = build_scope_filter_expr("partition_x", scope_field="partition_id")
        assert result == 'partition_id == "partition_x"'

    def test_no_scope_returns_existing_expr(self) -> None:
        """Test that None scope returns existing expression."""
        result = build_scope_filter_expr(None, existing_expr='category == "news"')
        assert result == 'category == "news"'

    def test_no_scope_on_empty_expr(self) -> None:
        """Test that None scope on empty expression returns empty string."""
        result = build_scope_filter_expr(None)
        assert result == ""

    def test_complex_existing_expr(self) -> None:
        """Test with complex existing expression."""
        existing = '(category == "news" OR category == "blog") AND author != null'
        result = build_scope_filter_expr("tenant_a", existing_expr=existing)
        assert result == f'({existing}) and namespace == "tenant_a"'
