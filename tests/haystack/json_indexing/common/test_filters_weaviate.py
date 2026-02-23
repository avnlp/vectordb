"""Tests for Weaviate filter builder.

Tests cover:
- Simple equality filters
- All comparison operators ($eq, $ne, $gt, $gte, $lt, $lte)
- Multiple conditions combined with AND
- Empty/None filters
- Various data types
"""

from unittest.mock import MagicMock, patch

from vectordb.haystack.json_indexing.common.filters.weaviate import (
    build_weaviate_filter,
)


class TestBuildWeaviateFilter:
    """Tests for build_weaviate_filter function."""

    def test_none_filter(self) -> None:
        """Test that None returns None."""
        result = build_weaviate_filter(None)
        assert result is None

    def test_empty_dict_filter(self) -> None:
        """Test that empty dict returns None."""
        result = build_weaviate_filter({})
        assert result is None

    def test_simple_equality_filter(self) -> None:
        """Test simple equality filter without operator."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = (
                mock_filter_instance
            )

            filters = {"category": "science"}
            result = build_weaviate_filter(filters)

            mock_filter.by_property.assert_called_once_with("category")
            mock_filter.by_property.return_value.equal.assert_called_once_with(
                "science"
            )
            assert result == mock_filter_instance

    def test_eq_operator_filter(self) -> None:
        """Test $eq operator filter."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = (
                mock_filter_instance
            )

            filters = {"status": {"$eq": "active"}}
            result = build_weaviate_filter(filters)

            mock_filter.by_property.assert_called_once_with("status")
            mock_filter.by_property.return_value.equal.assert_called_once_with("active")
            assert result == mock_filter_instance

    def test_ne_operator_filter(self) -> None:
        """Test $ne operator filter."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.not_equal.return_value = (
                mock_filter_instance
            )

            filters = {"status": {"$ne": "inactive"}}
            result = build_weaviate_filter(filters)

            mock_filter.by_property.assert_called_once_with("status")
            mock_filter.by_property.return_value.not_equal.assert_called_once_with(
                "inactive"
            )
            assert result == mock_filter_instance

    def test_gt_operator_filter(self) -> None:
        """Test $gt operator filter."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.greater_than.return_value = (
                mock_filter_instance
            )

            filters = {"age": {"$gt": 25}}
            result = build_weaviate_filter(filters)

            mock_filter.by_property.assert_called_once_with("age")
            mock_filter.by_property.return_value.greater_than.assert_called_once_with(
                25
            )
            assert result == mock_filter_instance

    def test_gte_operator_filter(self) -> None:
        """Test $gte operator filter."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.greater_or_equal.return_value = (
                mock_filter_instance
            )

            filters = {"score": {"$gte": 80}}
            result = build_weaviate_filter(filters)

            mock_filter.by_property.assert_called_once_with("score")
            mock_filter.by_property.return_value.greater_or_equal.assert_called_once_with(
                80
            )
            assert result == mock_filter_instance

    def test_lt_operator_filter(self) -> None:
        """Test $lt operator filter."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.less_than.return_value = (
                mock_filter_instance
            )

            filters = {"price": {"$lt": 100}}
            result = build_weaviate_filter(filters)

            mock_filter.by_property.assert_called_once_with("price")
            mock_filter.by_property.return_value.less_than.assert_called_once_with(100)
            assert result == mock_filter_instance

    def test_lte_operator_filter(self) -> None:
        """Test $lte operator filter."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.less_or_equal.return_value = (
                mock_filter_instance
            )

            filters = {"quantity": {"$lte": 50}}
            result = build_weaviate_filter(filters)

            mock_filter.by_property.assert_called_once_with("quantity")
            mock_filter.by_property.return_value.less_or_equal.assert_called_once_with(
                50
            )
            assert result == mock_filter_instance

    def test_multiple_conditions_with_and(self) -> None:
        """Test filter with multiple conditions combined with AND."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter1 = MagicMock()
            mock_filter2 = MagicMock()
            mock_combined = MagicMock()

            mock_filter.by_property.side_effect = [
                MagicMock(equal=MagicMock(return_value=mock_filter1)),
                MagicMock(greater_or_equal=MagicMock(return_value=mock_filter2)),
            ]
            mock_filter1.__and__ = MagicMock(return_value=mock_combined)

            filters = {
                "category": "science",
                "score": {"$gte": 80},
            }
            result = build_weaviate_filter(filters)

            # Should combine filters with AND
            mock_filter1.__and__.assert_called_once_with(mock_filter2)
            assert result == mock_combined

    def test_unsupported_operator_skipped(self) -> None:
        """Test that unsupported operators are skipped."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter.by_property.return_value.equal.return_value = MagicMock()

            # Has both $eq (supported) and $in (unsupported)
            filters = {
                "type": {"$eq": "premium"},
                "status": {"$in": ["active", "pending"]},
            }
            build_weaviate_filter(filters)

            # Should only create filter for $eq
            mock_filter.by_property.assert_called_once_with("type")

    def test_numeric_values(self) -> None:
        """Test filters with numeric values."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = (
                mock_filter_instance
            )

            filters = {"count": 42}
            build_weaviate_filter(filters)

            mock_filter.by_property.return_value.equal.assert_called_once_with(42)

    def test_boolean_values(self) -> None:
        """Test filters with boolean values."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.equal.return_value = (
                mock_filter_instance
            )

            filters = {"active": True}
            build_weaviate_filter(filters)

            mock_filter.by_property.return_value.equal.assert_called_once_with(True)

    def test_float_values(self) -> None:
        """Test filters with float values."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter.by_property.return_value.greater_than.return_value = (
                mock_filter_instance
            )

            filters = {"rating": {"$gt": 4.5}}
            build_weaviate_filter(filters)

            mock_filter.by_property.return_value.greater_than.assert_called_once_with(
                4.5
            )

    def test_three_conditions_chained(self) -> None:
        """Test filter with three conditions properly chained."""
        with patch(
            "vectordb.haystack.json_indexing.common.filters.weaviate.Filter"
        ) as mock_filter:
            mock_f1 = MagicMock()
            mock_f2 = MagicMock()
            mock_f3 = MagicMock()
            mock_combined1 = MagicMock()
            mock_combined2 = MagicMock()

            mock_filter.by_property.side_effect = [
                MagicMock(equal=MagicMock(return_value=mock_f1)),
                MagicMock(greater_than=MagicMock(return_value=mock_f2)),
                MagicMock(less_than=MagicMock(return_value=mock_f3)),
            ]
            mock_f1.__and__ = MagicMock(return_value=mock_combined1)
            mock_combined1.__and__ = MagicMock(return_value=mock_combined2)

            filters = {
                "category": "science",
                "score": {"$gt": 50},
                "age": {"$lt": 100},
            }
            result = build_weaviate_filter(filters)

            assert result == mock_combined2
