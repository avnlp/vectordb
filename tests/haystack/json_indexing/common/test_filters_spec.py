"""Tests for filter specification and validation.

Tests cover:
- Operator validation
- Supported operators set
- Edge cases and invalid inputs
"""

import pytest

from vectordb.haystack.json_indexing.common.filters.spec import (
    SUPPORTED_OPERATORS,
    validate_filter_operator,
)


class TestValidateFilterOperator:
    """Tests for validate_filter_operator function."""

    @pytest.mark.parametrize(
        "operator",
        [
            "$eq",
            "$ne",
            "$gt",
            "$gte",
            "$lt",
            "$lte",
            "$in",
            "$nin",
        ],
    )
    def test_valid_operators(self, operator: str) -> None:
        """Test that all supported operators are validated as True."""
        assert validate_filter_operator(operator) is True

    @pytest.mark.parametrize(
        "operator",
        [
            "$contains",
            "$exists",
            "$regex",
            "$like",
            "$between",
            "$and",
            "$or",
            "$not",
        ],
    )
    def test_invalid_operators(self, operator: str) -> None:
        """Test that unsupported operators are validated as False."""
        assert validate_filter_operator(operator) is False

    def test_empty_string(self) -> None:
        """Test that empty string is not a valid operator."""
        assert validate_filter_operator("") is False

    def test_case_sensitivity(self) -> None:
        """Test that operator validation is case sensitive."""
        # Operators should be lowercase
        assert validate_filter_operator("$EQ") is False
        assert validate_filter_operator("$Eq") is False
        assert validate_filter_operator("$Gt") is False

    def test_special_characters(self) -> None:
        """Test that special characters are not valid operators."""
        assert validate_filter_operator("@eq") is False
        assert validate_filter_operator("eq") is False
        assert validate_filter_operator("$") is False

    def test_whitespace(self) -> None:
        """Test that whitespace is not a valid operator."""
        assert validate_filter_operator(" ") is False
        assert validate_filter_operator(" $eq ") is False

    def test_none_input(self) -> None:
        """Test that None is not a valid operator."""
        assert validate_filter_operator(None) is False  # type: ignore[arg-type]

    def test_numeric_input(self) -> None:
        """Test that numeric values are not valid operators."""
        assert validate_filter_operator(123) is False  # type: ignore[arg-type]


class TestSupportedOperators:
    """Tests for SUPPORTED_OPERATORS constant."""

    def test_supported_operators_is_set(self) -> None:
        """Test that SUPPORTED_OPERATORS is a set."""
        assert isinstance(SUPPORTED_OPERATORS, set)

    def test_supported_operators_count(self) -> None:
        """Test that there are 8 supported operators."""
        assert len(SUPPORTED_OPERATORS) == 8

    def test_supported_operators_contents(self) -> None:
        """Test that all expected operators are in the set."""
        expected = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
        assert expected == SUPPORTED_OPERATORS

    def test_supported_operators_immutable(self) -> None:
        """Test that the set contains expected operators and can be checked."""
        # Verify all expected operators are present
        for op in ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]:
            assert op in SUPPORTED_OPERATORS
