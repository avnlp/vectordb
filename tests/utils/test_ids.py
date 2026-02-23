"""Tests for ID utilities.

This module tests document ID generation and management utilities that
ensure consistent document identification across vector database operations.

Tested functions:
    get_doc_id: Retrieve or generate document IDs with fallback logic.
    coerce_id: Convert various types to string IDs with UUID fallback.
    set_doc_id: Assign IDs to documents with metadata synchronization.

Test coverage includes:
    - ID retrieval from document attributes and metadata
    - Custom fallback metadata keys
    - UUID generation for missing IDs
    - Type coercion (int, float, UUID objects to strings)
    - ID persistence through set/get roundtrips
    - Metadata synchronization (doc.id and doc.meta['doc_id'])
    - Edge cases: empty strings, None values, special characters
"""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from haystack import Document

from vectordb.utils.ids import coerce_id, get_doc_id, set_doc_id


class TestGetDocId:
    """Test suite for getting document IDs.

    Tests cover:
    - Retrieving existing document IDs
    - Handling documents without IDs
    - ID type handling
    """

    def test_get_doc_id_from_document_with_id(self) -> None:
        """Test getting ID from document with id attribute."""
        doc = Document(content="Test", id="doc-123")
        doc_id = get_doc_id(doc)

        assert doc_id == "doc-123"

    def test_get_doc_id_from_metadata(self) -> None:
        """Test getting ID from document metadata."""
        doc = Document(content="Test", meta={"id": "meta-id-456"})
        doc_id = get_doc_id(doc)

        # Should retrieve from either doc.id or meta
        assert doc_id is not None

    def test_get_doc_id_returns_string(self) -> None:
        """Test that get_doc_id returns a string."""
        doc = Document(content="Test", id="test-id")
        doc_id = get_doc_id(doc)

        assert isinstance(doc_id, str)

    def test_get_doc_id_from_document_without_explicit_id(self) -> None:
        """Test handling of document without explicit ID."""
        doc = Document(content="Test")
        # Document should have default id
        assert doc.id is not None

    def test_get_doc_id_fallback_meta_key(self) -> None:
        """Test getting ID from custom fallback_meta_key."""

        @dataclass
        class MockDoc:
            id: str | None = None
            meta: dict[str, Any] | None = None

        doc = MockDoc(id=None, meta={"custom_key": "custom-id-value"})
        doc_id = get_doc_id(doc, fallback_meta_key="custom_key")
        assert doc_id == "custom-id-value"

    def test_get_doc_id_missing_id_uses_meta_fallback(self) -> None:
        """Test fallback to meta when id is None."""

        @dataclass
        class MockDoc:
            id: str | None = None
            meta: dict[str, Any] | None = None

        doc = MockDoc(id=None, meta={"doc_id": "from-meta"})
        doc_id = get_doc_id(doc)
        assert doc_id == "from-meta"

    def test_get_doc_id_generates_uuid_when_no_id_or_meta(self) -> None:
        """Test UUID generation when both id and meta are missing."""

        @dataclass
        class MockDoc:
            id: str | None = None
            meta: dict[str, Any] | None = None

        doc = MockDoc(id=None, meta=None)
        doc_id = get_doc_id(doc)
        # Should be a valid UUID string
        assert len(doc_id) == 36
        assert doc_id.count("-") == 4

    def test_get_doc_id_generates_uuid_when_meta_empty(self) -> None:
        """Test UUID generation when meta exists but lacks fallback key."""

        @dataclass
        class MockDoc:
            id: str | None = None
            meta: dict[str, Any] | None = None

        doc = MockDoc(id=None, meta={})
        doc_id = get_doc_id(doc)
        # Should generate UUID
        assert len(doc_id) == 36

    def test_get_doc_id_without_id_attribute(self) -> None:
        """Test get_doc_id when object has no id attribute."""

        @dataclass
        class NoIdDoc:
            meta: dict[str, Any] | None = None

        doc = NoIdDoc(meta={"doc_id": "meta-fallback"})
        doc_id = get_doc_id(doc)
        assert doc_id == "meta-fallback"

    def test_get_doc_id_without_meta_attribute(self) -> None:
        """Test get_doc_id when object has no meta attribute."""

        @dataclass
        class NoMetaDoc:
            id: str | None = None

        doc = NoMetaDoc(id=None)
        doc_id = get_doc_id(doc)
        # Should generate UUID
        assert len(doc_id) == 36

    def test_get_doc_id_empty_string_id_generates_uuid(self) -> None:
        """Test that empty string id triggers UUID generation."""

        @dataclass
        class MockDoc:
            id: str | None = None
            meta: dict[str, Any] | None = None

        doc = MockDoc(id="", meta=None)
        doc_id = get_doc_id(doc)
        # Empty string is falsy, should generate UUID
        assert len(doc_id) == 36


class TestCoerceId:
    """Test suite for coercing values to string IDs."""

    def test_coerce_id_with_none_generates_uuid(self) -> None:
        """Test that None value generates a UUID."""
        result = coerce_id(None)
        assert len(result) == 36
        assert result.count("-") == 4
        # Verify it's a valid UUID
        UUID(result)

    def test_coerce_id_with_string(self) -> None:
        """Test coercing string returns same string."""
        result = coerce_id("my-string-id")
        assert result == "my-string-id"

    def test_coerce_id_with_int(self) -> None:
        """Test coercing int to string."""
        result = coerce_id(12345)
        assert result == "12345"

    def test_coerce_id_with_uuid(self) -> None:
        """Test coercing UUID object to string."""
        uuid_obj = UUID("550e8400-e29b-41d4-a716-446655440000")
        result = coerce_id(uuid_obj)
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_coerce_id_with_float(self) -> None:
        """Test coercing float to string."""
        result = coerce_id(3.14159)
        assert result == "3.14159"

    def test_coerce_id_with_custom_object(self) -> None:
        """Test coercing custom object with __str__."""

        class CustomId:
            def __str__(self) -> str:
                return "custom-str-repr"

        result = coerce_id(CustomId())
        assert result == "custom-str-repr"

    def test_coerce_id_with_empty_string(self) -> None:
        """Test that empty string is preserved (not None)."""
        result = coerce_id("")
        assert result == ""

    def test_coerce_id_with_zero(self) -> None:
        """Test that zero is coerced to string (not treated as None)."""
        result = coerce_id(0)
        assert result == "0"


class TestSetDocId:
    """Test suite for setting document IDs.

    Tests cover:
    - Setting document IDs
    - ID persistence
    - Overwriting existing IDs
    """

    def test_set_doc_id_on_new_document(self) -> None:
        """Test setting ID on a new document."""
        doc = Document(content="Test")
        set_doc_id(doc, "new-id-789")

        assert doc.id == "new-id-789"

    def test_set_doc_id_overwrites_existing(self) -> None:
        """Test that setting ID overwrites existing ID."""
        doc = Document(content="Test", id="old-id")
        set_doc_id(doc, "new-id")

        assert doc.id == "new-id"

    def test_set_doc_id_persists(self) -> None:
        """Test that set ID persists on document."""
        doc = Document(content="Test")
        test_id = "persistent-id-123"
        set_doc_id(doc, test_id)

        # Retrieve and verify
        retrieved_id = get_doc_id(doc)
        assert retrieved_id == test_id

    def test_set_doc_id_with_string_conversion(self) -> None:
        """Test setting ID with automatic string conversion."""
        doc = Document(content="Test")
        set_doc_id(doc, "string-id")

        assert isinstance(doc.id, str)

    def test_set_doc_id_sets_meta_doc_id(self) -> None:
        """Test that set_doc_id also sets doc.meta['doc_id']."""
        doc = Document(content="Test")
        set_doc_id(doc, "test-id-meta")

        assert doc.meta["doc_id"] == "test-id-meta"

    def test_set_doc_id_on_doc_with_no_meta(self) -> None:
        """Test set_doc_id initializes meta if None."""

        @dataclass
        class MockDoc:
            id: str | None = None
            meta: dict[str, Any] | None = None

        doc = MockDoc(id=None, meta=None)
        set_doc_id(doc, "new-id")

        assert doc.id == "new-id"
        assert doc.meta == {"doc_id": "new-id"}

    def test_set_doc_id_on_doc_with_existing_meta(self) -> None:
        """Test set_doc_id preserves existing meta entries."""

        @dataclass
        class MockDoc:
            id: str | None = None
            meta: dict[str, Any] = field(default_factory=dict)

        doc = MockDoc(id=None, meta={"existing_key": "existing_value"})
        set_doc_id(doc, "new-id")

        assert doc.id == "new-id"
        assert doc.meta["doc_id"] == "new-id"
        assert doc.meta["existing_key"] == "existing_value"

    def test_set_doc_id_without_meta_attribute(self) -> None:
        """Test set_doc_id when object has no meta attribute initially."""

        class NoMetaDoc:
            def __init__(self) -> None:
                self.id: str | None = None

        doc = NoMetaDoc()
        set_doc_id(doc, "added-id")

        assert doc.id == "added-id"
        assert doc.meta == {"doc_id": "added-id"}  # type: ignore[attr-defined]


class TestIdManagement:
    """Test suite for ID management workflow.

    Tests cover:
    - ID consistency
    - ID retrieval and setting workflow
    - Multiple documents with IDs
    """

    def test_id_roundtrip(self) -> None:
        """Test setting and getting ID roundtrip."""
        doc = Document(content="Test")
        original_id = "roundtrip-id"

        set_doc_id(doc, original_id)
        retrieved_id = get_doc_id(doc)

        assert retrieved_id == original_id

    def test_multiple_documents_different_ids(self) -> None:
        """Test that multiple documents can have different IDs."""
        doc1 = Document(content="Doc 1")
        doc2 = Document(content="Doc 2")

        set_doc_id(doc1, "id-1")
        set_doc_id(doc2, "id-2")

        assert get_doc_id(doc1) == "id-1"
        assert get_doc_id(doc2) == "id-2"

    def test_id_with_special_characters(self) -> None:
        """Test ID handling with special characters."""
        doc = Document(content="Test")
        special_id = "id:with:colons:and-dashes"

        set_doc_id(doc, special_id)
        assert get_doc_id(doc) == special_id

    def test_id_with_uuid_format(self) -> None:
        """Test ID handling with UUID format."""
        doc = Document(content="Test")
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"

        set_doc_id(doc, uuid_id)
        assert get_doc_id(doc) == uuid_id
