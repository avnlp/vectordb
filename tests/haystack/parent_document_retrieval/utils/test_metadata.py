"""Tests for metadata utilities in parent document retrieval."""

import pytest

from vectordb.haystack.parent_document_retrieval.utils.metadata import (
    create_child_metadata,
    create_parent_metadata,
    get_level_from_metadata,
    get_parent_id_from_metadata,
)


class TestCreateParentMetadata:
    """Test suite for create_parent_metadata function."""

    def test_create_parent_metadata_required_only(self) -> None:
        """Test creating parent metadata with only required parameters."""
        result = create_parent_metadata(doc_idx=0, parent_idx=1)

        expected = {
            "level": 1,
            "doc_idx": 0,
            "parent_idx": 1,
            "source_id": None,
        }

        assert result == expected

    def test_create_parent_metadata_with_source_id(self) -> None:
        """Test creating parent metadata with source_id."""
        result = create_parent_metadata(
            doc_idx=2,
            parent_idx=0,
            source_id="source_123",
        )

        expected = {
            "level": 1,
            "doc_idx": 2,
            "parent_idx": 0,
            "source_id": "source_123",
        }

        assert result == expected

    def test_create_parent_metadata_with_extra_metadata(
        self, sample_extra_metadata: dict
    ) -> None:
        """Test creating parent metadata with extra metadata."""
        result = create_parent_metadata(
            doc_idx=1,
            parent_idx=2,
            source_id="source_456",
            extra_metadata=sample_extra_metadata,
        )

        assert result["level"] == 1
        assert result["doc_idx"] == 1
        assert result["parent_idx"] == 2
        assert result["source_id"] == "source_456"
        assert result["custom_field"] == "custom_value"
        assert result["priority"] == 1
        assert result["tags"] == ["test", "sample"]

    def test_create_parent_metadata_with_empty_extra_metadata(self) -> None:
        """Test creating parent metadata with empty extra_metadata dict."""
        result = create_parent_metadata(
            doc_idx=0,
            parent_idx=0,
            extra_metadata={},
        )

        expected = {
            "level": 1,
            "doc_idx": 0,
            "parent_idx": 0,
            "source_id": None,
        }

        assert result == expected

    def test_create_parent_metadata_none_extra_metadata(self) -> None:
        """Test creating parent metadata with None extra_metadata."""
        result = create_parent_metadata(
            doc_idx=0,
            parent_idx=0,
            extra_metadata=None,
        )

        expected = {
            "level": 1,
            "doc_idx": 0,
            "parent_idx": 0,
            "source_id": None,
        }

        assert result == expected

    def test_create_parent_metadata_negative_indices(self) -> None:
        """Test creating parent metadata with negative indices."""
        result = create_parent_metadata(doc_idx=-1, parent_idx=-2)

        assert result["doc_idx"] == -1
        assert result["parent_idx"] == -2
        assert result["level"] == 1

    def test_create_parent_metadata_large_indices(self) -> None:
        """Test creating parent metadata with large indices."""
        result = create_parent_metadata(doc_idx=999999, parent_idx=888888)

        assert result["doc_idx"] == 999999
        assert result["parent_idx"] == 888888
        assert result["level"] == 1

    def test_create_parent_metadata_empty_string_source_id(self) -> None:
        """Test creating parent metadata with empty string source_id."""
        result = create_parent_metadata(
            doc_idx=0,
            parent_idx=0,
            source_id="",
        )

        assert result["source_id"] == ""

    def test_create_parent_metadata_conflicting_extra_metadata(self) -> None:
        """Test creating parent metadata with conflicting extra_metadata keys."""
        result = create_parent_metadata(
            doc_idx=0,
            parent_idx=0,
            source_id="original",
            extra_metadata={"source_id": "conflicting", "level": 999},
        )

        # Extra metadata should override base metadata
        assert result["source_id"] == "conflicting"
        assert result["level"] == 999
        assert result["doc_idx"] == 0
        assert result["parent_idx"] == 0

    def test_create_parent_metadata_various_extra_types(self) -> None:
        """Test creating parent metadata with various data types in extra_metadata."""
        extra_metadata = {
            "string_field": "test",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "list_field": [1, 2, 3],
            "dict_field": {"nested": "value"},
            "none_field": None,
        }

        result = create_parent_metadata(
            doc_idx=0,
            parent_idx=0,
            extra_metadata=extra_metadata,
        )

        for key, value in extra_metadata.items():
            assert result[key] == value


class TestCreateChildMetadata:
    """Test suite for create_child_metadata function."""

    def test_create_child_metadata_required_only(self) -> None:
        """Test creating child metadata with only required parameters."""
        result = create_child_metadata(
            parent_id="parent_123",
            doc_idx=0,
            parent_idx=1,
            child_idx=2,
        )

        expected = {
            "level": 2,
            "parent_id": "parent_123",
            "doc_idx": 0,
            "parent_idx": 1,
            "child_idx": 2,
            "source_id": None,
        }

        assert result == expected

    def test_create_child_metadata_with_source_id(self) -> None:
        """Test creating child metadata with source_id."""
        result = create_child_metadata(
            parent_id="parent_456",
            doc_idx=1,
            parent_idx=0,
            child_idx=0,
            source_id="source_789",
        )

        expected = {
            "level": 2,
            "parent_id": "parent_456",
            "doc_idx": 1,
            "parent_idx": 0,
            "child_idx": 0,
            "source_id": "source_789",
        }

        assert result == expected

    def test_create_child_metadata_with_extra_metadata(
        self, sample_extra_metadata: dict
    ) -> None:
        """Test creating child metadata with extra metadata."""
        result = create_child_metadata(
            parent_id="parent_abc",
            doc_idx=2,
            parent_idx=3,
            child_idx=1,
            source_id="source_xyz",
            extra_metadata=sample_extra_metadata,
        )

        assert result["level"] == 2
        assert result["parent_id"] == "parent_abc"
        assert result["doc_idx"] == 2
        assert result["parent_idx"] == 3
        assert result["child_idx"] == 1
        assert result["source_id"] == "source_xyz"
        assert result["custom_field"] == "custom_value"
        assert result["priority"] == 1
        assert result["tags"] == ["test", "sample"]

    def test_create_child_metadata_empty_parent_id(self) -> None:
        """Test creating child metadata with empty parent_id."""
        result = create_child_metadata(
            parent_id="",
            doc_idx=0,
            parent_idx=0,
            child_idx=0,
        )

        assert result["parent_id"] == ""

    def test_create_child_metadata_special_characters_parent_id(self) -> None:
        """Test creating child metadata with special characters in parent_id."""
        special_parent_id = "parent_123!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = create_child_metadata(
            parent_id=special_parent_id,
            doc_idx=0,
            parent_idx=0,
            child_idx=0,
        )

        assert result["parent_id"] == special_parent_id

    def test_create_child_metadata_negative_indices(self) -> None:
        """Test creating child metadata with negative indices."""
        result = create_child_metadata(
            parent_id="parent_test",
            doc_idx=-1,
            parent_idx=-2,
            child_idx=-3,
        )

        assert result["doc_idx"] == -1
        assert result["parent_idx"] == -2
        assert result["child_idx"] == -3
        assert result["level"] == 2

    def test_create_child_metadata_conflicting_extra_metadata(self) -> None:
        """Test creating child metadata with conflicting extra_metadata keys."""
        result = create_child_metadata(
            parent_id="parent_original",
            doc_idx=0,
            parent_idx=0,
            child_idx=0,
            source_id="source_original",
            extra_metadata={
                "parent_id": "parent_conflicting",
                "source_id": "source_conflicting",
                "level": 999,
            },
        )

        # Extra metadata should override base metadata
        assert result["parent_id"] == "parent_conflicting"
        assert result["source_id"] == "source_conflicting"
        assert result["level"] == 999
        assert result["doc_idx"] == 0
        assert result["parent_idx"] == 0
        assert result["child_idx"] == 0

    @pytest.mark.parametrize(
        "parent_id", ["parent_1", "parent_abc", "123", "parent-with-dash"]
    )
    def test_create_child_metadata_various_parent_ids(self, parent_id: str) -> None:
        """Test create_child_metadata with various parent_id formats."""
        result = create_child_metadata(
            parent_id=parent_id,
            doc_idx=0,
            parent_idx=0,
            child_idx=0,
        )

        assert result["parent_id"] == parent_id
        assert result["level"] == 2


class TestGetLevelFromMetadata:
    """Test suite for get_level_from_metadata function."""

    def test_get_level_from_metadata_parent(self, sample_metadata: dict) -> None:
        """Test getting level from parent metadata."""
        result = get_level_from_metadata(sample_metadata)

        assert result == 1

    def test_get_level_from_metadata_child(self) -> None:
        """Test getting level from child metadata."""
        child_metadata = {"level": 2, "parent_id": "parent_123"}

        result = get_level_from_metadata(child_metadata)

        assert result == 2

    def test_get_level_from_metadata_missing_level(self) -> None:
        """Test getting level when level key is missing."""
        metadata_without_level = {"parent_id": "parent_123", "doc_idx": 0}

        result = get_level_from_metadata(metadata_without_level)

        assert result is None

    def test_get_level_from_metadata_empty_dict(self) -> None:
        """Test getting level from empty metadata dict."""
        result = get_level_from_metadata({})

        assert result is None

    def test_get_level_from_metadata_none(self) -> None:
        """Test getting level from None metadata."""
        result = get_level_from_metadata(None)

        assert result is None

    def test_get_level_from_metadata_string_level(self) -> None:
        """Test getting level when level is a string."""
        metadata_with_string_level = {"level": "1", "parent_id": "parent_123"}

        result = get_level_from_metadata(metadata_with_string_level)

        assert result == "1"

    def test_get_level_from_metadata_various_levels(self) -> None:
        """Test getting level with various level values."""
        test_cases = [
            ({"level": 0}, 0),
            ({"level": 1}, 1),
            ({"level": 2}, 2),
            ({"level": 3}, 3),
            ({"level": -1}, -1),
            ({"level": 999}, 999),
        ]

        for metadata, expected in test_cases:
            result = get_level_from_metadata(metadata)
            assert result == expected

    def test_get_level_from_metadata_with_other_fields(self) -> None:
        """Test getting level when metadata has many other fields."""
        complex_metadata = {
            "level": 1,
            "parent_id": "parent_123",
            "doc_idx": 0,
            "parent_idx": 0,
            "source_id": "source_456",
            "custom_field": "custom_value",
            "tags": ["tag1", "tag2"],
        }

        result = get_level_from_metadata(complex_metadata)

        assert result == 1


class TestGetParentIdFromMetadata:
    """Test suite for get_parent_id_from_metadata function."""

    def test_get_parent_id_from_metadata_with_parent_id(
        self, sample_metadata: dict
    ) -> None:
        """Test getting parent_id when it exists."""
        result = get_parent_id_from_metadata(sample_metadata)

        assert result == "parent_123"

    def test_get_parent_id_from_metadata_missing_parent_id(self) -> None:
        """Test getting parent_id when key is missing."""
        metadata_without_parent_id = {"level": 2, "doc_idx": 0}

        result = get_parent_id_from_metadata(metadata_without_parent_id)

        assert result is None

    def test_get_parent_id_from_metadata_empty_dict(self) -> None:
        """Test getting parent_id from empty metadata dict."""
        result = get_parent_id_from_metadata({})

        assert result is None

    def test_get_parent_id_from_metadata_none(self) -> None:
        """Test getting parent_id from None metadata."""
        result = get_parent_id_from_metadata(None)

        assert result is None

    def test_get_parent_id_from_metadata_non_string_parent_id(self) -> None:
        """Test getting parent_id when it's not a string."""
        metadata_with_int_parent_id = {"parent_id": 123, "level": 2}

        result = get_parent_id_from_metadata(metadata_with_int_parent_id)

        assert result == 123

    def test_get_parent_id_from_metadata_empty_string(self) -> None:
        """Test getting parent_id when it's an empty string."""
        metadata_with_empty_parent_id = {"parent_id": "", "level": 2}

        result = get_parent_id_from_metadata(metadata_with_empty_parent_id)

        assert result == ""

    def test_get_parent_id_from_metadata_various_formats(self) -> None:
        """Test getting parent_id with various formats."""
        test_cases = [
            ({"parent_id": "parent_123"}, "parent_123"),
            ({"parent_id": "123"}, "123"),
            ({"parent_id": "parent-with-dash"}, "parent-with-dash"),
            ({"parent_id": "parent_with_underscore"}, "parent_with_underscore"),
            ({"parent_id": "parent.with.dots"}, "parent.with.dots"),
        ]

        for metadata, expected in test_cases:
            result = get_parent_id_from_metadata(metadata)
            assert result == expected

    def test_get_parent_id_from_metadata_with_other_fields(self) -> None:
        """Test getting parent_id when metadata has many other fields."""
        complex_metadata = {
            "parent_id": "parent_complex_123",
            "level": 2,
            "doc_idx": 1,
            "parent_idx": 0,
            "child_idx": 0,
            "source_id": "source_456",
            "custom_field": "custom_value",
            "tags": ["tag1", "tag2"],
        }

        result = get_parent_id_from_metadata(complex_metadata)

        assert result == "parent_complex_123"

    @pytest.mark.parametrize("parent_id", [None, "", "parent_1", "123", 456])
    def test_get_parent_id_from_metadata_parameterized(self, parent_id) -> None:
        """Test get_parent_id_from_metadata with various parent_id values."""
        metadata = {"parent_id": parent_id, "level": 2}

        result = get_parent_id_from_metadata(metadata)

        assert result == parent_id
