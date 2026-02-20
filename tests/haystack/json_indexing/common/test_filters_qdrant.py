"""Tests for Qdrant filter builder.

Tests cover:
- Simple equality filters
- Operator-based filters ($eq, $ne, $gt, $gte, $lt, $lte)
- Multiple conditions
- Empty/None filters
- Complex nested filters
"""

from unittest.mock import MagicMock, patch

from vectordb.haystack.json_indexing.common.filters.qdrant import build_qdrant_filter


class TestBuildQdrantFilter:
    """Tests for build_qdrant_filter function."""

    def test_none_filter(self) -> None:
        """Test that None returns None."""
        result = build_qdrant_filter(None)
        assert result is None

    def test_empty_dict_filter(self) -> None:
        """Test that empty dict returns None."""
        result = build_qdrant_filter({})
        assert result is None

    def test_simple_equality_filter(self) -> None:
        """Test simple equality filter without operator."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ) as mock_field_condition,
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.MatchValue"
            ) as mock_match_value,
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.Filter"
            ) as mock_filter,
        ):
            mock_match = MagicMock()
            mock_match_value.return_value = mock_match
            mock_condition = MagicMock()
            mock_field_condition.return_value = mock_condition
            mock_filter_instance = MagicMock()
            mock_filter.return_value = mock_filter_instance

            filters = {"category": "science"}
            result = build_qdrant_filter(filters)

            mock_match_value.assert_called_once_with(value="science")
            mock_field_condition.assert_called_once_with(
                key="category", match=mock_match
            )
            mock_filter.assert_called_once_with(must=[mock_condition])
            assert result == mock_filter_instance

    def test_eq_operator_filter(self) -> None:
        """Test $eq operator filter."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ) as mock_field_condition,
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.MatchValue"
            ) as mock_match_value,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            mock_match = MagicMock()
            mock_match_value.return_value = mock_match
            mock_condition = MagicMock()
            mock_field_condition.return_value = mock_condition

            filters = {"status": {"$eq": "active"}}
            build_qdrant_filter(filters)

            mock_match_value.assert_called_once_with(value="active")
            mock_field_condition.assert_called_once_with(key="status", match=mock_match)

    def test_ne_operator_skipped(self) -> None:
        """Test that $ne operator is skipped (not natively supported)."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ) as mock_field_condition,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            # Only $ne filter should result in no conditions
            filters = {"status": {"$ne": "inactive"}}
            result = build_qdrant_filter(filters)

            # Should return None since $ne is skipped
            assert result is None
            mock_field_condition.assert_not_called()

    def test_gt_operator_filter(self) -> None:
        """Test $gt operator filter."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ) as mock_field_condition,
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.Range"
            ) as mock_range,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            mock_range_instance = MagicMock()
            mock_range.return_value = mock_range_instance
            mock_condition = MagicMock()
            mock_field_condition.return_value = mock_condition

            filters = {"age": {"$gt": 25}}
            build_qdrant_filter(filters)

            mock_range.assert_called_once_with(gt=25)
            mock_field_condition.assert_called_once_with(
                key="age", range=mock_range_instance
            )

    def test_gte_operator_filter(self) -> None:
        """Test $gte operator filter."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ),
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.Range"
            ) as mock_range,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            mock_range_instance = MagicMock()
            mock_range.return_value = mock_range_instance

            filters = {"score": {"$gte": 80}}
            build_qdrant_filter(filters)

            mock_range.assert_called_once_with(gte=80)

    def test_lt_operator_filter(self) -> None:
        """Test $lt operator filter."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ),
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.Range"
            ) as mock_range,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            mock_range_instance = MagicMock()
            mock_range.return_value = mock_range_instance

            filters = {"price": {"$lt": 100}}
            build_qdrant_filter(filters)

            mock_range.assert_called_once_with(lt=100)

    def test_lte_operator_filter(self) -> None:
        """Test $lte operator filter."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ),
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.Range"
            ) as mock_range,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            mock_range_instance = MagicMock()
            mock_range.return_value = mock_range_instance

            filters = {"quantity": {"$lte": 50}}
            build_qdrant_filter(filters)

            mock_range.assert_called_once_with(lte=50)

    def test_multiple_conditions(self) -> None:
        """Test filter with multiple conditions."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ) as mock_field_condition,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.MatchValue"),
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Range"),
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.Filter"
            ) as mock_filter,
        ):
            mock_condition = MagicMock()
            mock_field_condition.return_value = mock_condition
            mock_filter_instance = MagicMock()
            mock_filter.return_value = mock_filter_instance

            filters = {
                "category": "science",
                "score": {"$gte": 80},
            }
            result = build_qdrant_filter(filters)

            # Should have 2 conditions
            assert mock_field_condition.call_count == 2
            mock_filter.assert_called_once_with(must=[mock_condition, mock_condition])
            assert result == mock_filter_instance

    def test_numeric_values(self) -> None:
        """Test filters with numeric values."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ),
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.MatchValue"
            ) as mock_match_value,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            filters = {"count": 42}
            build_qdrant_filter(filters)

            mock_match_value.assert_called_once_with(value=42)

    def test_boolean_values(self) -> None:
        """Test filters with boolean values."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ),
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.MatchValue"
            ) as mock_match_value,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            filters = {"active": True}
            build_qdrant_filter(filters)

            mock_match_value.assert_called_once_with(value=True)

    def test_only_ne_operator_returns_none(self) -> None:
        """Test that filter with only $ne returns None."""
        filters = {"status": {"$ne": "deleted"}}
        result = build_qdrant_filter(filters)
        assert result is None

    def test_mixed_operators_with_ne(self) -> None:
        """Test filter with mix of supported and unsupported operators."""
        with (
            patch(
                "vectordb.haystack.json_indexing.common.filters.qdrant.FieldCondition"
            ) as mock_field_condition,
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.MatchValue"),
            patch("vectordb.haystack.json_indexing.common.filters.qdrant.Filter"),
        ):
            mock_condition = MagicMock()
            mock_field_condition.return_value = mock_condition

            # Has both $eq (supported) and $ne (skipped)
            filters = {
                "type": {"$eq": "premium"},
                "status": {"$ne": "banned"},
            }
            result = build_qdrant_filter(filters)

            # Should only create one condition (for $eq)
            assert mock_field_condition.call_count == 1
            assert result is not None
