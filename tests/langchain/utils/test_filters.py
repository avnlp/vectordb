"""Tests for document filtering utilities.

This module tests the DocumentFilter class which provides metadata-based
filtering for LangChain Document collections. Filters reduce result sets
based on document attributes without requiring re-embedding.

DocumentFilter Methods:
    filter_by_metadata: Filter by single metadata key with operators
    filter_by_predicate: Filter by custom predicate function
    filter_by_metadata_json: Filter by nested JSON path in metadata
    exclude_by_metadata: Exclude documents matching metadata criteria

Supported Operators:
    equals, contains, startswith, endswith, gt, lt, gte, lte, in, not_in

Test Classes:
    TestFilterByMetadata: Operator-based filtering
    TestFilterByPredicate: Custom predicate function filtering
    TestFilterByMetadataJson: Nested JSON path filtering
    TestExcludeByMetadata: Exclusion-based filtering
    TestDocumentFilterEdgeCases: None values, numeric strings, mixed types
"""

import pytest
from langchain_core.documents import Document

from vectordb.langchain.utils.filters import DocumentFilter


class TestFilterByMetadata:
    """Tests for DocumentFilter.filter_by_metadata operator-based filtering.

    Validates filtering documents by metadata key/value pairs using various
    comparison operators. Supports string matching, numeric comparisons,
    and collection membership tests.

    Default Operator:
        equals - exact match on metadata value
    """

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Doc 1", metadata={"category": "A", "priority": 1}),
            Document(page_content="Doc 2", metadata={"category": "B", "priority": 2}),
            Document(page_content="Doc 3", metadata={"category": "A", "priority": 3}),
            Document(page_content="Doc 4", metadata={"category": "C", "priority": 1}),
            Document(page_content="Doc 5", metadata={"category": "B", "priority": 2}),
        ]

    def test_filter_equals(self, sample_documents):
        """Test filtering with equals operator."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="category", value="A", operator="equals"
        )
        assert len(result) == 2
        for doc in result:
            assert doc.metadata["category"] == "A"

    def test_filter_contains(self, sample_documents):
        """Test filtering with contains operator."""
        docs = [
            Document(
                page_content="Python tutorial", metadata={"title": "Python Guide"}
            ),
            Document(page_content="Java tutorial", metadata={"title": "Java Basics"}),
            Document(page_content="Python tricks", metadata={"title": "Python Tricks"}),
        ]
        result = DocumentFilter.filter_by_metadata(
            docs, key="title", value="Python", operator="contains"
        )
        assert len(result) == 2

    def test_filter_startswith(self, sample_documents):
        """Test filtering with startswith operator."""
        docs = [
            Document(page_content="Doc", metadata={"name": "alpha_one"}),
            Document(page_content="Doc", metadata={"name": "alpha_two"}),
            Document(page_content="Doc", metadata={"name": "beta_one"}),
        ]
        result = DocumentFilter.filter_by_metadata(
            docs, key="name", value="alpha", operator="startswith"
        )
        assert len(result) == 2

    def test_filter_endswith(self, sample_documents):
        """Test filtering with endswith operator."""
        docs = [
            Document(page_content="Doc", metadata={"name": "file_one_txt"}),
            Document(page_content="Doc", metadata={"name": "file_two_txt"}),
            Document(page_content="Doc", metadata={"name": "file_three_md"}),
        ]
        result = DocumentFilter.filter_by_metadata(
            docs, key="name", value="_txt", operator="endswith"
        )
        assert len(result) == 2

    def test_filter_gt(self, sample_documents):
        """Test filtering with greater than operator."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="priority", value=1, operator="gt"
        )
        assert len(result) == 3  # priorities 2, 3, 2 (Doc2, Doc3, Doc5)
        for doc in result:
            assert doc.metadata["priority"] > 1

    def test_filter_lt(self, sample_documents):
        """Test filtering with less than operator."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="priority", value=3, operator="lt"
        )
        assert len(result) == 4  # priorities 1, 2, 1, 2 (Doc1, Doc2, Doc4, Doc5)
        for doc in result:
            assert doc.metadata["priority"] < 3

    def test_filter_gte(self, sample_documents):
        """Test filtering with greater than or equal operator."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="priority", value=2, operator="gte"
        )
        assert len(result) == 3  # priorities 2, 3, 2 (Doc2, Doc3, Doc5)
        for doc in result:
            assert doc.metadata["priority"] >= 2

    def test_filter_lte(self, sample_documents):
        """Test filtering with less than or equal operator."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="priority", value=2, operator="lte"
        )
        assert len(result) == 4  # priorities 1, 2, 1, 2
        for doc in result:
            assert doc.metadata["priority"] <= 2

    def test_filter_in(self, sample_documents):
        """Test filtering with in operator."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="category", value=["A", "B"], operator="in"
        )
        assert len(result) == 4  # A, B, A, B

    def test_filter_not_in(self, sample_documents):
        """Test filtering with not_in operator."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="category", value=["A", "B"], operator="not_in"
        )
        assert len(result) == 1  # Only C
        assert result[0].metadata["category"] == "C"

    def test_filter_key_not_present(self, sample_documents):
        """Test filtering with key not present in any document."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="nonexistent", value="test", operator="equals"
        )
        assert result == []

    def test_default_operator_equals(self, sample_documents):
        """Test that default operator is equals."""
        result = DocumentFilter.filter_by_metadata(
            sample_documents, key="category", value="A"
        )
        assert len(result) == 2

    def test_unknown_operator_raises_error(self, sample_documents):
        """Test that unknown operator raises ValueError."""
        with pytest.raises(ValueError, match="Unknown operator"):
            DocumentFilter.filter_by_metadata(
                sample_documents, key="category", value="A", operator="unknown"
            )

    def test_empty_document_list(self):
        """Test filtering with empty document list."""
        result = DocumentFilter.filter_by_metadata([], key="category", value="A")
        assert result == []


class TestFilterByPredicate:
    """Tests for DocumentFilter.filter_by_predicate custom function filtering.

    Validates filtering using arbitrary Python predicate functions. Predicates
    receive a Document object and return True to include, False to exclude.
    Useful for complex filtering logic not expressible with operators.
    """

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Short", metadata={"length": 5}),
            Document(page_content="Medium length text", metadata={"length": 17}),
            Document(
                page_content="Very long document content here", metadata={"length": 33}
            ),
        ]

    def test_filter_with_predicate(self, sample_documents):
        """Test filtering with custom predicate."""
        result = DocumentFilter.filter_by_predicate(
            sample_documents, predicate=lambda doc: len(doc.page_content) > 10
        )
        assert len(result) == 2

    def test_filter_no_matches(self, sample_documents):
        """Test filtering with predicate that matches nothing."""
        result = DocumentFilter.filter_by_predicate(
            sample_documents, predicate=lambda doc: "nonexistent" in doc.page_content
        )
        assert result == []

    def test_filter_all_match(self, sample_documents):
        """Test filtering with predicate that matches all."""
        result = DocumentFilter.filter_by_predicate(
            sample_documents, predicate=lambda doc: len(doc.page_content) > 0
        )
        assert len(result) == 3

    def test_filter_with_metadata_predicate(self, sample_documents):
        """Test filtering with metadata-based predicate."""
        result = DocumentFilter.filter_by_predicate(
            sample_documents, predicate=lambda doc: doc.metadata.get("length", 0) > 10
        )
        assert len(result) == 2

    def test_empty_document_list(self):
        """Test filtering with empty document list."""
        result = DocumentFilter.filter_by_predicate([], predicate=lambda doc: True)
        assert result == []


class TestFilterByMetadataJson:
    """Tests for DocumentFilter.filter_by_metadata_json nested path filtering.

    Validates filtering by dot-notation JSON paths within metadata. Supports
    deeply nested structures like "author.name" or "a.b.c.value". Uses same
    operators as filter_by_metadata.

    Path Resolution:
        Paths are split by "." and traversed through nested dicts. If any
        intermediate path is not a dict or doesn't exist, document is excluded.
    """

    def test_filter_nested_equals(self):
        """Test filtering with nested JSON path equals."""
        docs = [
            Document(page_content="Doc 1", metadata={"author": {"name": "Alice"}}),
            Document(page_content="Doc 2", metadata={"author": {"name": "Bob"}}),
            Document(page_content="Doc 3", metadata={"author": {"name": "Alice"}}),
        ]
        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="author.name", value="Alice", operator="equals"
        )
        assert len(result) == 2

    def test_filter_nested_deeper_path(self):
        """Test filtering with deeper nested path."""
        docs = [
            Document(page_content="Doc 1", metadata={"a": {"b": {"c": {"value": 1}}}}),
            Document(page_content="Doc 2", metadata={"a": {"b": {"c": {"value": 2}}}}),
        ]
        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="a.b.c.value", value=1, operator="equals"
        )
        assert len(result) == 1

    def test_filter_nested_contains(self):
        """Test filtering with nested path and contains operator."""
        docs = [
            Document(
                page_content="Doc 1", metadata={"info": {"email": "alice@test.com"}}
            ),
            Document(
                page_content="Doc 2", metadata={"info": {"email": "bob@other.com"}}
            ),
        ]
        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="info.email", value="test.com", operator="contains"
        )
        assert len(result) == 1

    def test_filter_path_not_found(self):
        """Test filtering when path doesn't exist."""
        docs = [
            Document(page_content="Doc 1", metadata={"author": "Alice"}),
            Document(page_content="Doc 2", metadata={"author": "Bob"}),
        ]
        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="author.name", value="Alice", operator="equals"
        )
        assert result == []

    def test_filter_nested_gt(self):
        """Test filtering with nested path and greater than operator."""
        docs = [
            Document(page_content="Doc 1", metadata={"stats": {"score": 10}}),
            Document(page_content="Doc 2", metadata={"stats": {"score": 20}}),
            Document(page_content="Doc 3", metadata={"stats": {"score": 30}}),
        ]
        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="stats.score", value=15, operator="gt"
        )
        assert len(result) == 2

    def test_filter_nested_in(self):
        """Test filtering with nested path and in operator."""
        docs = [
            Document(page_content="Doc 1", metadata={"tags": {"primary": "A"}}),
            Document(page_content="Doc 2", metadata={"tags": {"primary": "B"}}),
            Document(page_content="Doc 3", metadata={"tags": {"primary": "C"}}),
        ]
        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="tags.primary", value=["A", "B"], operator="in"
        )
        assert len(result) == 2

    def test_filter_nested_non_dict_intermediate(self):
        """Test filtering when intermediate path is not a dict."""
        docs = [
            Document(page_content="Doc 1", metadata={"author": "Alice"}),
            Document(
                page_content="Doc 2",
                metadata={"author": {"name": "Alice", "role": "editor"}},
            ),
        ]

        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="author.name", value="Alice", operator="equals"
        )

        assert len(result) == 1
        assert result[0].metadata["author"]["name"] == "Alice"

    def test_empty_document_list(self):
        """Test filtering with empty document list."""
        result = DocumentFilter.filter_by_metadata_json(
            [], json_path="author.name", value="Alice", operator="equals"
        )
        assert result == []


class TestExcludeByMetadata:
    """Tests for DocumentFilter.exclude_by_metadata exclusion filtering.

    Validates exclusion of documents matching specific metadata criteria.
    Inverse of filter_by_metadata - returns documents that do NOT match.
    Useful for removing known-bad documents from result sets.
    """

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Doc 1", metadata={"category": "A"}),
            Document(page_content="Doc 2", metadata={"category": "B"}),
            Document(page_content="Doc 3", metadata={"category": "A"}),
            Document(page_content="Doc 4", metadata={"category": "C"}),
        ]

    def test_exclude_single_value(self, sample_documents):
        """Test excluding documents with specific metadata value."""
        result = DocumentFilter.exclude_by_metadata(
            sample_documents, key="category", value="A"
        )
        assert len(result) == 2  # Only B and C
        for doc in result:
            assert doc.metadata["category"] != "A"

    def test_exclude_all(self, sample_documents):
        """Test excluding all documents."""
        result = DocumentFilter.exclude_by_metadata(
            sample_documents, key="category", value="A"
        )
        # Exclude A, keep B and C
        assert len(result) == 2

    def test_exclude_none(self, sample_documents):
        """Test excluding value not present."""
        result = DocumentFilter.exclude_by_metadata(
            sample_documents, key="category", value="Z"
        )
        assert len(result) == 4  # All documents kept

    def test_exclude_missing_key(self, sample_documents):
        """Test excluding with missing key (no documents match)."""
        result = DocumentFilter.exclude_by_metadata(
            sample_documents, key="nonexistent", value="test"
        )
        assert len(result) == 4  # All documents kept since none have the key

    def test_exclude_preserves_remaining(self, sample_documents):
        """Test that excluded documents are properly removed."""
        result = DocumentFilter.exclude_by_metadata(
            sample_documents, key="category", value="A"
        )
        contents = [doc.page_content for doc in result]
        assert "Doc 1" not in contents  # Excluded
        assert "Doc 2" in contents  # Kept
        assert "Doc 3" not in contents  # Excluded
        assert "Doc 4" in contents  # Kept

    def test_empty_document_list(self):
        """Test excluding with empty document list."""
        result = DocumentFilter.exclude_by_metadata([], key="category", value="A")
        assert result == []


class TestDocumentFilterEdgeCases:
    """Edge case tests for DocumentFilter boundary conditions.

    Validates handling of unusual metadata values and types including None,
    numeric strings, empty metadata, mixed types in lists, and deeply nested
    paths that don't exist.

    Edge Cases Covered:
        - None metadata values
        - Numeric strings vs integers
        - Case-insensitive contains matching
        - Empty metadata dicts
        - Lists as metadata values
        - Missing deeply nested paths
    """

    def test_metadata_with_none_value(self):
        """Test filtering with None metadata value."""
        docs = [
            Document(page_content="Doc 1", metadata={"key": None}),
            Document(page_content="Doc 2", metadata={"key": "value"}),
        ]
        result = DocumentFilter.filter_by_metadata(
            docs, key="key", value=None, operator="equals"
        )
        assert len(result) == 1

    def test_metadata_with_numeric_string(self):
        """Test filtering when metadata value is numeric string."""
        docs = [
            Document(page_content="Doc 1", metadata={"count": "10"}),
            Document(page_content="Doc 2", metadata={"count": "20"}),
        ]
        result = DocumentFilter.filter_by_metadata(
            docs, key="count", value="10", operator="equals"
        )
        assert len(result) == 1

    def test_case_insensitive_contains(self):
        """Test case-insensitive contains operator."""
        docs = [
            Document(page_content="Doc 1", metadata={"text": "HELLO"}),
            Document(page_content="Doc 2", metadata={"text": "hello"}),
            Document(page_content="Doc 3", metadata={"text": "World"}),
        ]
        result = DocumentFilter.filter_by_metadata(
            docs, key="text", value="hello", operator="contains"
        )
        assert len(result) == 2

    def test_empty_metadata(self):
        """Test filtering with empty metadata."""
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={"key": "value"}),
        ]
        result = DocumentFilter.filter_by_metadata(
            docs, key="key", value="value", operator="equals"
        )
        assert len(result) == 1

    def test_mixed_types_in_list(self):
        """Test filtering with list containing mixed types."""
        docs = [
            Document(page_content="Doc 1", metadata={"items": [1, 2, 3]}),
            Document(page_content="Doc 2", metadata={"items": [4, 5, 6]}),
        ]
        # The in operator should work
        result = DocumentFilter.filter_by_metadata(
            docs, key="items", value=[1, 2, 3], operator="equals"
        )
        assert len(result) == 1

    def test_deeply_nested_missing_path(self):
        """Test filtering with deeply nested path that doesn't exist."""
        docs = [
            Document(page_content="Doc 1", metadata={"a": {"b": 1}}),
            Document(page_content="Doc 2", metadata={"a": {"c": 2}}),
        ]
        result = DocumentFilter.filter_by_metadata_json(
            docs, json_path="a.b.c.d", value=1, operator="equals"
        )
        assert result == []
