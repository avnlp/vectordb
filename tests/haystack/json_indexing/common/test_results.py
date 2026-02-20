"""Tests for result normalization utilities.

Tests cover:
- Normalization of search results
- Handling missing fields
- Empty results
- Various data types in results
"""

from vectordb.haystack.json_indexing.common.results import normalize_search_results


class TestNormalizeSearchResults:
    """Tests for normalize_search_results function."""

    def test_normalize_single_result(self) -> None:
        """Test normalizing a single search result."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "content": "This is test content",
                "metadata": {"source": "wiki", "author": "test"},
            }
        ]

        result = normalize_search_results(raw_results)

        assert len(result) == 1
        assert result[0]["id"] == "doc1"
        assert result[0]["score"] == 0.95
        assert result[0]["content"] == "This is test content"
        assert result[0]["metadata"] == {"source": "wiki", "author": "test"}

    def test_normalize_multiple_results(self) -> None:
        """Test normalizing multiple search results."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "content": "First document",
                "metadata": {"source": "wiki"},
            },
            {
                "id": "doc2",
                "score": 0.87,
                "content": "Second document",
                "metadata": {"source": "blog"},
            },
            {
                "id": "doc3",
                "score": 0.72,
                "content": "Third document",
                "metadata": {"source": "paper"},
            },
        ]

        result = normalize_search_results(raw_results)

        assert len(result) == 3
        assert result[0]["id"] == "doc1"
        assert result[1]["id"] == "doc2"
        assert result[2]["id"] == "doc3"
        assert result[0]["score"] == 0.95
        assert result[1]["score"] == 0.87
        assert result[2]["score"] == 0.72

    def test_missing_id_field(self) -> None:
        """Test normalizing result with missing id field."""
        raw_results = [
            {
                "score": 0.95,
                "content": "Test content",
                "metadata": {"source": "wiki"},
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["id"] is None
        assert result[0]["score"] == 0.95
        assert result[0]["content"] == "Test content"

    def test_missing_score_field(self) -> None:
        """Test normalizing result with missing score field."""
        raw_results = [
            {
                "id": "doc1",
                "content": "Test content",
                "metadata": {"source": "wiki"},
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["id"] == "doc1"
        assert result[0]["score"] is None
        assert result[0]["content"] == "Test content"

    def test_missing_content_field(self) -> None:
        """Test normalizing result with missing content field."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "metadata": {"source": "wiki"},
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["id"] == "doc1"
        assert result[0]["score"] == 0.95
        assert result[0]["content"] is None

    def test_missing_metadata_field(self) -> None:
        """Test normalizing result with missing metadata field."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "content": "Test content",
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["metadata"] == {}

    def test_empty_metadata(self) -> None:
        """Test normalizing result with empty metadata."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "content": "Test content",
                "metadata": {},
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["metadata"] == {}

    def test_empty_results_list(self) -> None:
        """Test normalizing empty results list."""
        raw_results: list[dict] = []

        result = normalize_search_results(raw_results)

        assert result == []

    def test_extra_fields_ignored(self) -> None:
        """Test that extra fields are not included in normalized result."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "content": "Test content",
                "metadata": {"source": "wiki"},
                "extra_field": "should_be_ignored",
                "another_extra": 123,
            }
        ]

        result = normalize_search_results(raw_results)

        assert "extra_field" not in result[0]
        assert "another_extra" not in result[0]
        assert set(result[0].keys()) == {"id", "score", "content", "metadata"}

    def test_various_data_types(self) -> None:
        """Test normalizing results with various data types."""
        raw_results = [
            {
                "id": 123,  # numeric id
                "score": 0.95,
                "content": "Test content",
                "metadata": {
                    "count": 42,
                    "active": True,
                    "ratio": 3.14,
                    "tags": ["a", "b", "c"],
                },
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["id"] == 123
        assert result[0]["metadata"]["count"] == 42
        assert result[0]["metadata"]["active"] is True
        assert result[0]["metadata"]["ratio"] == 3.14
        assert result[0]["metadata"]["tags"] == ["a", "b", "c"]

    def test_none_values_in_fields(self) -> None:
        """Test normalizing results with explicit None values."""
        raw_results = [
            {
                "id": None,
                "score": None,
                "content": None,
                "metadata": None,
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["id"] is None
        assert result[0]["score"] is None
        assert result[0]["content"] is None
        # None metadata is preserved as None (not converted to empty dict)
        assert result[0]["metadata"] is None

    def test_string_score(self) -> None:
        """Test normalizing result with string score."""
        raw_results = [
            {
                "id": "doc1",
                "score": "0.95",  # string instead of float
                "content": "Test content",
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["score"] == "0.95"

    def test_nested_metadata(self) -> None:
        """Test normalizing result with deeply nested metadata."""
        raw_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "content": "Test content",
                "metadata": {
                    "level1": {
                        "level2": {
                            "level3": "deep_value",
                        },
                    },
                },
            }
        ]

        result = normalize_search_results(raw_results)

        assert result[0]["metadata"]["level1"]["level2"]["level3"] == "deep_value"
