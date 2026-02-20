"""Tests for DocumentFilter utility class."""

import pytest
from haystack import Document

from vectordb.haystack.utils.filters import DocumentFilter


class TestDocumentFilter:
    """Tests for DocumentFilter class."""

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Sample documents for testing."""
        return [
            Document(content="Doc 1", meta={"category": "science", "score": 85}),
            Document(content="Doc 2", meta={"category": "history", "score": 70}),
            Document(content="Doc 3", meta={"category": "science", "score": 95}),
            Document(content="Doc 4", meta={"category": "math", "score": 60}),
        ]

    def test_normalize_empty(self) -> None:
        """Test normalize returns None for empty filters."""
        assert DocumentFilter.normalize(None) is None
        assert DocumentFilter.normalize({}) is None

    def test_normalize_passes_through(self) -> None:
        """Test normalize returns filter dict unchanged."""
        filters = {"category": "science"}
        result = DocumentFilter.normalize(filters)
        assert result == filters

    def test_apply_no_filters(self, sample_documents: list[Document]) -> None:
        """Test apply returns all documents when no filters."""
        result = DocumentFilter.apply(sample_documents, None)
        assert len(result) == 4

        result = DocumentFilter.apply(sample_documents, {})
        assert len(result) == 4

    def test_apply_simple_equality(self, sample_documents: list[Document]) -> None:
        """Test simple equality filter."""
        result = DocumentFilter.apply(sample_documents, {"category": "science"})
        assert len(result) == 2
        assert all(doc.meta["category"] == "science" for doc in result)

    def test_apply_eq_operator(self, sample_documents: list[Document]) -> None:
        """Test $eq operator."""
        result = DocumentFilter.apply(
            sample_documents, {"category": {"$eq": "history"}}
        )
        assert len(result) == 1
        assert result[0].content == "Doc 2"

    def test_apply_ne_operator(self, sample_documents: list[Document]) -> None:
        """Test $ne operator."""
        result = DocumentFilter.apply(
            sample_documents, {"category": {"$ne": "science"}}
        )
        assert len(result) == 2
        assert all(doc.meta["category"] != "science" for doc in result)

    def test_apply_gt_operator(self, sample_documents: list[Document]) -> None:
        """Test $gt operator."""
        result = DocumentFilter.apply(sample_documents, {"score": {"$gt": 80}})
        assert len(result) == 2
        assert all(doc.meta["score"] > 80 for doc in result)

    def test_apply_gte_operator(self, sample_documents: list[Document]) -> None:
        """Test $gte operator."""
        result = DocumentFilter.apply(sample_documents, {"score": {"$gte": 85}})
        assert len(result) == 2
        assert all(doc.meta["score"] >= 85 for doc in result)

    def test_apply_lt_operator(self, sample_documents: list[Document]) -> None:
        """Test $lt operator."""
        result = DocumentFilter.apply(sample_documents, {"score": {"$lt": 75}})
        assert len(result) == 2
        assert all(doc.meta["score"] < 75 for doc in result)

    def test_apply_lte_operator(self, sample_documents: list[Document]) -> None:
        """Test $lte operator."""
        result = DocumentFilter.apply(sample_documents, {"score": {"$lte": 70}})
        assert len(result) == 2
        assert all(doc.meta["score"] <= 70 for doc in result)

    def test_apply_in_operator(self, sample_documents: list[Document]) -> None:
        """Test $in operator."""
        result = DocumentFilter.apply(
            sample_documents, {"category": {"$in": ["science", "math"]}}
        )
        assert len(result) == 3

    def test_apply_nin_operator(self, sample_documents: list[Document]) -> None:
        """Test $nin operator."""
        result = DocumentFilter.apply(
            sample_documents, {"category": {"$nin": ["science", "math"]}}
        )
        assert len(result) == 1
        assert result[0].meta["category"] == "history"

    def test_apply_contains_operator(self) -> None:
        """Test $contains operator."""
        docs = [
            Document(content="Hello world", meta={"text": "hello there"}),
            Document(content="Goodbye", meta={"text": "see you later"}),
        ]
        result = DocumentFilter.apply(docs, {"text": {"$contains": "hello"}})
        assert len(result) == 1
        assert result[0].content == "Hello world"

    def test_apply_multiple_conditions(self, sample_documents: list[Document]) -> None:
        """Test multiple filter conditions (AND logic)."""
        result = DocumentFilter.apply(
            sample_documents, {"category": "science", "score": {"$gte": 90}}
        )
        assert len(result) == 1
        assert result[0].content == "Doc 3"

    def test_apply_missing_field(self) -> None:
        """Test filtering on missing field."""
        docs = [
            Document(content="Has field", meta={"field": "value"}),
            Document(content="No field", meta={}),
        ]
        result = DocumentFilter.apply(docs, {"field": "value"})
        assert len(result) == 1
        assert result[0].content == "Has field"

    def test_apply_none_value_with_ne(self) -> None:
        """Test $ne with None values."""
        docs = [
            Document(content="Has field", meta={"field": "value"}),
            Document(content="No field", meta={}),
        ]
        result = DocumentFilter.apply(docs, {"field": {"$ne": "value"}})
        assert len(result) == 1
        assert result[0].content == "No field"
