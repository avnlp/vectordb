"""Tests for vectordb_pipeline_type utilities."""

from typing import Any

import pytest

from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    ChromaFilterExpressionBuilder,
    FilterCondition,
    FilteredQueryResult,
    FilterField,
    FilterSpec,
    MilvusFilterExpressionBuilder,
    PineconeFilterExpressionBuilder,
    QdrantFilterExpressionBuilder,
    SelectivityAnalyzer,
    TimingMetrics,
    WeaviateFilterExpressionBuilder,
    parse_filter_from_config,
    validate_filter_config,
)


def make_invalid_condition(field: str, operator: str, value: Any) -> FilterCondition:
    """Create a FilterCondition without running validation."""
    condition = object.__new__(FilterCondition)
    condition.field = field
    condition.operator = operator
    condition.value = value
    return condition


class TestFilterField:
    """Tests for FilterField dataclass."""

    def test_filter_field_creation(self):
        """Test basic FilterField creation."""
        field = FilterField(
            name="category",
            type="string",
            operators=["eq", "ne", "contains"],
            description="Document category",
        )
        assert field.name == "category"
        assert field.type == "string"
        assert field.operators == ["eq", "ne", "contains"]
        assert field.indexed is True
        assert field.nullable is False

    def test_filter_field_defaults(self):
        """Test FilterField default values."""
        field = FilterField(
            name="test", type="string", operators=[], description="Test field"
        )
        assert field.indexed is True
        assert field.nullable is False


class TestFilterCondition:
    """Tests for FilterCondition dataclass."""

    def test_filter_condition_valid_operators(self):
        """Test FilterCondition with valid operators."""
        valid_operators = [
            "eq",
            "ne",
            "gt",
            "gte",
            "lt",
            "lte",
            "in",
            "contains",
            "range",
        ]
        for op in valid_operators:
            condition = FilterCondition(field="test", operator=op, value="value")
            assert condition.operator == op

    def test_filter_condition_invalid_operator(self):
        """Test FilterCondition with invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid operator"):
            FilterCondition(field="test", operator="invalid", value="value")

    def test_filter_condition_equality(self):
        """Test FilterCondition equality."""
        cond1 = FilterCondition(field="category", operator="eq", value="tech")
        cond2 = FilterCondition(field="category", operator="eq", value="tech")
        assert cond1 == cond2


class TestFilterSpec:
    """Tests for FilterSpec dataclass."""

    def test_filter_spec_creation(self):
        """Test basic FilterSpec creation."""
        conditions = [
            FilterCondition(field="category", operator="eq", value="tech"),
            FilterCondition(field="year", operator="gte", value=2020),
        ]
        spec = FilterSpec(conditions=conditions)
        assert len(spec.conditions) == 2

    def test_filter_spec_empty(self):
        """Test FilterSpec with no conditions."""
        spec = FilterSpec(conditions=[])
        assert len(spec.conditions) == 0


class TestTimingMetrics:
    """Tests for TimingMetrics dataclass."""

    def test_timing_metrics_creation(self):
        """Test basic TimingMetrics creation."""
        metrics = TimingMetrics(
            pre_filter_ms=10.5,
            vector_search_ms=25.0,
            total_ms=35.5,
            num_candidates=100,
            num_total_docs=1000,
        )
        assert metrics.pre_filter_ms == 10.5
        assert metrics.selectivity == 0.1

    def test_timing_metrics_selectivity_zero_total(self):
        """Test selectivity when total docs is zero."""
        metrics = TimingMetrics(
            pre_filter_ms=0,
            vector_search_ms=0,
            total_ms=0,
            num_candidates=0,
            num_total_docs=0,
        )
        assert metrics.selectivity == 0.0

    def test_timing_metrics_selectivity_full_match(self):
        """Test selectivity when all documents match."""
        metrics = TimingMetrics(
            pre_filter_ms=0,
            vector_search_ms=0,
            total_ms=0,
            num_candidates=500,
            num_total_docs=500,
        )
        assert metrics.selectivity == 1.0


class TestFilteredQueryResult:
    """Tests for FilteredQueryResult dataclass."""

    def test_filtered_query_result_creation(self):
        """Test basic FilteredQueryResult creation."""
        from haystack import Document

        doc = Document(content="test", meta={"key": "value"})
        result = FilteredQueryResult(
            document=doc,
            relevance_score=0.95,
            rank=1,
            filter_matched=True,
            timing=None,
        )
        assert result.document == doc
        assert result.relevance_score == 0.95
        assert result.rank == 1
        assert result.filter_matched is True


class TestFilterExpressionBuilderValidation:
    """Tests for base FilterExpressionBuilder validation."""

    def test_validate_condition_unknown_field(self):
        """Test validate_condition raises for unknown field."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        condition = FilterCondition(field="unknown", operator="eq", value="test")
        with pytest.raises(ValueError, match="Field not in schema"):
            builder.validate_condition(condition)

    def test_validate_condition_unsupported_operator(self):
        """Test validate_condition raises for unsupported operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        condition = FilterCondition(field="category", operator="gt", value=1)
        with pytest.raises(ValueError, match="Operator gt not supported"):
            builder.validate_condition(condition)


class TestMilvusFilterExpressionBuilder:
    """Tests for MilvusFilterExpressionBuilder class."""

    def test_build_empty_spec(self):
        """Test building empty filter spec."""
        builder = MilvusFilterExpressionBuilder({})
        spec = FilterSpec(conditions=[])
        result = builder.build(spec)
        assert result == ""

    def test_build_eq_operator(self):
        """Test building eq operator."""
        schema = {
            "category": FilterField(
                name="category",
                type="string",
                operators=["eq", "ne"],
                description="Category",
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        result = builder.build(spec)
        assert 'metadata["category"] == "tech"' in result

    def test_build_gt_operator(self):
        """Test building gt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["gt", "gte"], description="Year"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="gt", value=2020)]
        )
        result = builder.build(spec)
        assert 'metadata["year"] > 2020' in result

    def test_build_ne_operator(self):
        """Test building ne operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["ne"], description="Category"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="ne", value="tech")]
        )
        result = builder.build(spec)
        assert 'metadata["category"] != "tech"' in result

    def test_build_lt_operator(self):
        """Test building lt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lt"], description="Year"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lt", value=2020)]
        )
        result = builder.build(spec)
        assert 'metadata["year"] < 2020' in result

    def test_build_lte_operator(self):
        """Test building lte operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lte"], description="Year"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lte", value=2020)]
        )
        result = builder.build(spec)
        assert 'metadata["year"] <= 2020' in result

    def test_build_in_operator(self):
        """Test building in operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["in"], description="Category"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="in", value=["a", "b", "c"])
            ]
        )
        result = builder.build(spec)
        assert 'metadata["category"] in ["a", "b", "c"]' in result

    def test_build_in_operator_requires_list(self):
        """Test building in operator with non-list value raises error."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["in"], description="Category"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="in", value="single")
            ]
        )
        with pytest.raises(ValueError, match="'in' operator requires list"):
            builder.build(spec)

    def test_build_contains_operator(self):
        """Test building contains operator."""
        schema = {
            "text": FilterField(
                name="text", type="string", operators=["contains"], description="Text"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="text", operator="contains", value="term")
            ]
        )
        result = builder.build(spec)
        assert 'metadata["text"] like "%term%"' in result

    def test_build_range_operator(self):
        """Test building range operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="year", operator="range", value=(2020, 2024))
            ]
        )
        result = builder.build(spec)
        assert "&&" in result

    def test_build_range_operator_invalid_value(self):
        """Test building range operator with invalid value raises error."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="range", value=2020)]
        )
        with pytest.raises(ValueError, match="'range' operator requires"):
            builder.build(spec)

    def test_build_range_operator_invalid_tuple_length(self):
        """Test building range operator with short tuple raises error."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="range", value=(2020,))]
        )
        with pytest.raises(ValueError, match="'range' operator requires"):
            builder.build(spec)

    def test_build_unsupported_operator(self):
        """Test building unsupported operator raises error."""
        builder = MilvusFilterExpressionBuilder({})
        condition = make_invalid_condition("category", "unsupported", "value")
        with pytest.raises(ValueError, match="Unsupported operator"):
            builder._build_condition(condition)

    def test_build_multiple_conditions(self):
        """Test building multiple conditions with AND."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            ),
            "year": FilterField(
                name="year", type="integer", operators=["gte"], description="Year"
            ),
        }
        builder = MilvusFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="eq", value="tech"),
                FilterCondition(field="year", operator="gte", value=2020),
            ]
        )
        result = builder.build(spec)
        assert " && " in result

    def test_format_value_string(self):
        """Test _format_value with string."""
        result = MilvusFilterExpressionBuilder._format_value("test")
        assert result == '"test"'

    def test_format_value_bool_true(self):
        """Test _format_value with True."""
        result = MilvusFilterExpressionBuilder._format_value(True)
        assert result == "true"

    def test_format_value_bool_false(self):
        """Test _format_value with False."""
        result = MilvusFilterExpressionBuilder._format_value(False)
        assert result == "false"

    def test_format_value_number(self):
        """Test _format_value with number."""
        result = MilvusFilterExpressionBuilder._format_value(42)
        assert result == "42"


class TestQdrantFilterExpressionBuilder:
    """Tests for QdrantFilterExpressionBuilder class."""

    def test_build_empty_spec(self):
        """Test building empty filter spec."""
        builder = QdrantFilterExpressionBuilder({})
        spec = FilterSpec(conditions=[])
        result = builder.build(spec)
        # Should return empty Filter with must=[] or None
        assert result is not None

    def test_build_eq_operator(self):
        """Test building eq operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        result = builder.build(spec)
        # Result should be a Filter object
        assert result is not None

    def test_build_ne_operator(self):
        """Test building ne operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["ne"], description="Category"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="ne", value="tech")]
        )
        result = builder.build(spec)
        assert result is not None

    def test_build_gt_operator(self):
        """Test building gt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["gt"], description="Year"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        condition = FilterCondition(field="year", operator="gt", value=2020)
        result = builder._build_condition(condition)
        assert result is not None

    def test_build_gte_operator(self):
        """Test building gte operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["gte"], description="Year"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        condition = FilterCondition(field="year", operator="gte", value=2020)
        result = builder._build_condition(condition)
        assert result is not None

    def test_build_lt_operator(self):
        """Test building lt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lt"], description="Year"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        condition = FilterCondition(field="year", operator="lt", value=2020)
        result = builder._build_condition(condition)
        assert result is not None

    def test_build_lte_operator(self):
        """Test building lte operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lte"], description="Year"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        condition = FilterCondition(field="year", operator="lte", value=2020)
        result = builder._build_condition(condition)
        assert result is not None

    def test_build_in_operator(self):
        """Test building in operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["in"], description="Category"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        condition = FilterCondition(field="category", operator="in", value=["a", "b"])
        result = builder._build_condition(condition)
        assert result is not None

    def test_build_contains_operator(self):
        """Test building contains operator."""
        schema = {
            "text": FilterField(
                name="text", type="string", operators=["contains"], description="Text"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        condition = FilterCondition(field="text", operator="contains", value="term")
        result = builder._build_condition(condition)
        assert result is not None

    def test_build_range_operator(self):
        """Test building range operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="year", operator="range", value=(2020, 2024))
            ]
        )
        result = builder.build(spec)
        assert result is not None

    def test_build_range_operator_invalid_value(self):
        """Test building range operator with invalid value raises error."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = QdrantFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="range", value="bad")]
        )
        with pytest.raises(ValueError, match="'range' requires"):
            builder.build(spec)

    def test_build_unsupported_operator(self):
        """Test building unsupported operator raises error."""
        builder = QdrantFilterExpressionBuilder({})
        condition = make_invalid_condition("category", "unsupported", "value")
        with pytest.raises(ValueError, match="Unsupported operator"):
            builder._build_condition(condition)


class TestPineconeFilterExpressionBuilder:
    """Tests for PineconeFilterExpressionBuilder class."""

    def test_build_empty_spec(self):
        """Test building empty filter spec."""
        builder = PineconeFilterExpressionBuilder({})
        spec = FilterSpec(conditions=[])
        result = builder.build(spec)
        assert result == {}

    def test_build_eq_operator(self):
        """Test building eq operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        result = builder.build(spec)
        assert result == {"category": {"$eq": "tech"}}

    def test_build_ne_operator(self):
        """Test building ne operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["ne"], description="Category"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="ne", value="tech")]
        )
        result = builder.build(spec)
        assert result == {"category": {"$ne": "tech"}}

    def test_build_gt_operator(self):
        """Test building gt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["gt"], description="Year"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="gt", value=2020)]
        )
        result = builder.build(spec)
        assert result == {"year": {"$gt": 2020}}

    def test_build_multiple_conditions(self):
        """Test building multiple conditions."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            ),
            "year": FilterField(
                name="year", type="integer", operators=["gte"], description="Year"
            ),
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="eq", value="tech"),
                FilterCondition(field="year", operator="gte", value=2020),
            ]
        )
        result = builder.build(spec)
        assert "$and" in result

    def test_build_in_operator(self):
        """Test building in operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["in"], description="Category"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="in", value=["a", "b"])
            ]
        )
        result = builder.build(spec)
        assert result == {"category": {"$in": ["a", "b"]}}

    def test_build_lt_operator(self):
        """Test building lt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lt"], description="Year"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lt", value=2020)]
        )
        result = builder.build(spec)
        assert result == {"year": {"$lt": 2020}}

    def test_build_lte_operator(self):
        """Test building lte operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lte"], description="Year"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lte", value=2020)]
        )
        result = builder.build(spec)
        assert result == {"year": {"$lte": 2020}}

    def test_build_contains_operator(self):
        """Test building contains operator."""
        schema = {
            "text": FilterField(
                name="text", type="string", operators=["contains"], description="Text"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="text", operator="contains", value="term")
            ]
        )
        result = builder.build(spec)
        assert result == {"text": {"$in": ["term"]}}

    def test_build_range_operator_invalid_value(self):
        """Test building range operator with invalid value raises error."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = PineconeFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="range", value=2020)]
        )
        with pytest.raises(ValueError, match="'range' requires"):
            builder.build(spec)

    def test_build_unsupported_operator(self):
        """Test building unsupported operator raises error."""
        builder = PineconeFilterExpressionBuilder({})
        condition = make_invalid_condition("category", "unsupported", "value")
        with pytest.raises(ValueError, match="Unsupported operator"):
            builder._build_condition(condition)


class TestWeaviateFilterExpressionBuilder:
    """Tests for WeaviateFilterExpressionBuilder class."""

    def test_build_empty_spec(self):
        """Test building empty filter spec."""
        builder = WeaviateFilterExpressionBuilder({})
        spec = FilterSpec(conditions=[])
        result = builder.build(spec)
        assert result == {}

    def test_build_eq_operator_string(self):
        """Test building eq operator with string."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        result = builder.build(spec)
        assert result["operator"] == "Equal"
        assert result["path"] == ["category"]
        assert result["valueString"] == "tech"

    def test_build_eq_operator_number(self):
        """Test building eq operator with number."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["eq"], description="Year"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="eq", value=2020)]
        )
        result = builder.build(spec)
        assert result["operator"] == "Equal"
        assert result["valueNumber"] == 2020

    def test_build_ne_operator(self):
        """Test building ne operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["ne"], description="Category"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="ne", value="tech")]
        )
        result = builder.build(spec)
        assert result["operator"] == "NotEqual"

    def test_build_gt_operator(self):
        """Test building gt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["gt"], description="Year"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="gt", value=2020)]
        )
        result = builder.build(spec)
        assert result["operator"] == "GreaterThan"

    def test_build_lt_operator(self):
        """Test building lt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lt"], description="Year"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lt", value=2020)]
        )
        result = builder.build(spec)
        assert result["operator"] == "LessThan"

    def test_build_lte_operator(self):
        """Test building lte operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lte"], description="Year"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lte", value=2020)]
        )
        result = builder.build(spec)
        assert result["operator"] == "LessThanEqual"
        assert result["valueNumber"] == 2020

    def test_build_in_operator_string_list(self):
        """Test building in operator with string list."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["in"], description="Category"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="in", value=["a", "b"])
            ]
        )
        result = builder.build(spec)
        assert result["operator"] == "ContainsAny"
        assert result["valueStringArray"] == ["a", "b"]

    def test_build_contains_operator(self):
        """Test building contains operator."""
        schema = {
            "category": FilterField(
                name="category",
                type="string",
                operators=["contains"],
                description="Category",
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="contains", value="test")
            ]
        )
        result = builder.build(spec)
        assert result["operator"] == "Like"

    def test_build_range_operator(self):
        """Test building range operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="year", operator="range", value=(2020, 2024))
            ]
        )
        result = builder.build(spec)
        assert result["operator"] == "And"
        assert len(result["operands"]) == 2

    def test_build_range_operator_invalid_value(self):
        """Test building range operator with invalid value raises error."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="range", value="bad")]
        )
        with pytest.raises(ValueError, match="'range' requires"):
            builder.build(spec)

    def test_build_unsupported_operator(self):
        """Test building unsupported operator raises error."""
        builder = WeaviateFilterExpressionBuilder({})
        condition = make_invalid_condition("category", "unsupported", "value")
        with pytest.raises(ValueError, match="Unsupported operator"):
            builder._build_condition(condition)

    def test_build_multiple_conditions(self):
        """Test building multiple conditions."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            ),
            "year": FilterField(
                name="year", type="integer", operators=["gte"], description="Year"
            ),
        }
        builder = WeaviateFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="eq", value="tech"),
                FilterCondition(field="year", operator="gte", value=2020),
            ]
        )
        result = builder.build(spec)
        assert result["operator"] == "And"
        assert len(result["operands"]) == 2


class TestChromaFilterExpressionBuilder:
    """Tests for ChromaFilterExpressionBuilder class."""

    def test_build_empty_spec(self):
        """Test building empty filter spec."""
        builder = ChromaFilterExpressionBuilder({})
        spec = FilterSpec(conditions=[])
        result = builder.build(spec)
        assert result == {}

    def test_build_eq_operator(self):
        """Test building eq operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        result = builder.build(spec)
        # Chroma eq is just the value directly
        assert result == {"category": "tech"}

    def test_build_ne_operator(self):
        """Test building ne operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["ne"], description="Category"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="ne", value="tech")]
        )
        result = builder.build(spec)
        assert result == {"category": {"$ne": "tech"}}

    def test_build_gt_operator(self):
        """Test building gt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["gt"], description="Year"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="gt", value=2020)]
        )
        result = builder.build(spec)
        assert result == {"year": {"$gt": 2020}}

    def test_build_gte_operator(self):
        """Test building gte operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["gte"], description="Year"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="gte", value=2020)]
        )
        result = builder.build(spec)
        assert result == {"year": {"$gte": 2020}}

    def test_build_lt_operator(self):
        """Test building lt operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lt"], description="Year"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lt", value=2020)]
        )
        result = builder.build(spec)
        assert result == {"year": {"$lt": 2020}}

    def test_build_lte_operator(self):
        """Test building lte operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["lte"], description="Year"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="lte", value=2020)]
        )
        result = builder.build(spec)
        assert result == {"year": {"$lte": 2020}}

    def test_build_in_operator(self):
        """Test building in operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["in"], description="Category"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="in", value=["a", "b"])
            ]
        )
        result = builder.build(spec)
        assert result == {"category": {"$in": ["a", "b"]}}

    def test_build_contains_operator(self):
        """Test building contains operator."""
        schema = {
            "category": FilterField(
                name="category",
                type="string",
                operators=["contains"],
                description="Category",
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="contains", value="test")
            ]
        )
        result = builder.build(spec)
        assert result == {"category": {"$contains": "test"}}

    def test_build_range_operator(self):
        """Test building range operator."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="year", operator="range", value=(2020, 2024))
            ]
        )
        result = builder.build(spec)
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_build_range_operator_invalid_value(self):
        """Test building range operator with invalid value raises error."""
        schema = {
            "year": FilterField(
                name="year", type="integer", operators=["range"], description="Year"
            )
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[FilterCondition(field="year", operator="range", value="bad")]
        )
        with pytest.raises(ValueError, match="'range' requires"):
            builder.build(spec)

    def test_build_unsupported_operator(self):
        """Test building unsupported operator raises error."""
        builder = ChromaFilterExpressionBuilder({})
        condition = make_invalid_condition("category", "unsupported", "value")
        with pytest.raises(ValueError, match="Unsupported operator"):
            builder._build_condition(condition)

    def test_build_multiple_conditions(self):
        """Test building multiple conditions."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            ),
            "year": FilterField(
                name="year", type="integer", operators=["gte"], description="Year"
            ),
        }
        builder = ChromaFilterExpressionBuilder(schema)
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="eq", value="tech"),
                FilterCondition(field="year", operator="gte", value=2020),
            ]
        )
        result = builder.build(spec)
        assert "$and" in result


class TestParseFilterFromConfig:
    """Tests for parse_filter_from_config function."""

    def test_parse_filter_from_config_valid(self):
        """Test parsing valid filter config."""
        config = {
            "metadata_filtering": {
                "test_filters": [
                    {
                        "name": "test_filter",
                        "conditions": [
                            {"field": "category", "operator": "eq", "value": "tech"},
                        ],
                    }
                ]
            }
        }
        result = parse_filter_from_config(config)
        assert len(result.conditions) == 1
        assert result.conditions[0].field == "category"
        assert result.conditions[0].operator == "eq"
        assert result.conditions[0].value == "tech"

    def test_parse_filter_from_config_multiple_conditions(self):
        """Test parsing config with multiple conditions."""
        config = {
            "metadata_filtering": {
                "test_filters": [
                    {
                        "conditions": [
                            {"field": "category", "operator": "eq", "value": "tech"},
                            {"field": "year", "operator": "gte", "value": 2020},
                        ],
                    }
                ]
            }
        }
        result = parse_filter_from_config(config)
        assert len(result.conditions) == 2

    def test_parse_filter_from_config_missing_test_filters(self):
        """Test parsing config without test_filters raises error."""
        config = {"metadata_filtering": {}}
        with pytest.raises(ValueError, match="No test_filters found"):
            parse_filter_from_config(config)

    def test_parse_filter_from_config_empty_test_filters(self):
        """Test parsing config with empty test_filters list."""
        config = {"metadata_filtering": {"test_filters": []}}
        with pytest.raises(ValueError, match="No test_filters found"):
            parse_filter_from_config(config)

    def test_parse_filter_from_config_non_list_test_filters(self):
        """Test parsing config with non-list test_filters raises error."""
        config = {"metadata_filtering": {"test_filters": "invalid"}}
        with pytest.raises(ValueError, match="test_filters must be non-empty list"):
            parse_filter_from_config(config)


class TestValidateFilterConfig:
    """Tests for validate_filter_config function."""

    def test_validate_filter_config_valid(self):
        """Test validating valid filter config."""
        schema = {
            "category": FilterField(
                name="category",
                type="string",
                operators=["eq", "ne"],
                description="Category",
            ),
            "year": FilterField(
                name="year",
                type="integer",
                operators=["gte", "lte"],
                description="Year",
            ),
        }
        config = {
            "metadata_filtering": {
                "test_filters": [
                    {
                        "name": "test",
                        "conditions": [
                            {"field": "category", "operator": "eq", "value": "tech"},
                        ],
                    }
                ]
            }
        }
        is_valid, errors = validate_filter_config(config, schema)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_filter_config_unknown_field(self):
        """Test validating config with unknown field."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        config = {
            "metadata_filtering": {
                "test_filters": [
                    {
                        "name": "test",
                        "conditions": [
                            {"field": "unknown", "operator": "eq", "value": "test"},
                        ],
                    }
                ]
            }
        }
        is_valid, errors = validate_filter_config(config, schema)
        assert is_valid is False
        assert any("Unknown field" in e for e in errors)

    def test_validate_filter_config_unsupported_operator(self):
        """Test validating config with unsupported operator."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        config = {
            "metadata_filtering": {
                "test_filters": [
                    {
                        "name": "test",
                        "conditions": [
                            {"field": "category", "operator": "gt", "value": "test"},
                        ],
                    }
                ]
            }
        }
        is_valid, errors = validate_filter_config(config, schema)
        assert is_valid is False
        assert any("Operator" in e and "not supported" in e for e in errors)

    def test_validate_filter_config_missing_test_filters(self):
        """Test validating config without test_filters."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        config = {"metadata_filtering": {}}
        is_valid, errors = validate_filter_config(config, schema)
        assert is_valid is False
        assert errors == ["No test_filters found in metadata_filtering section"]

    def test_validate_filter_config_non_list_test_filters(self):
        """Test validating config with non-list test_filters."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        config = {"metadata_filtering": {"test_filters": "invalid"}}
        is_valid, errors = validate_filter_config(config, schema)
        assert is_valid is False
        assert errors == ["test_filters must be non-empty list"]

    def test_validate_filter_config_empty_conditions(self):
        """Test validating config with empty conditions."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        config = {
            "metadata_filtering": {
                "test_filters": [{"name": "empty", "conditions": []}]
            }
        }
        is_valid, errors = validate_filter_config(config, schema)
        assert is_valid is False
        assert any("has no conditions" in error for error in errors)


class TestSelectivityAnalyzer:
    """Tests for SelectivityAnalyzer class."""

    def test_estimate_selectivity_empty_docs(self):
        """Test selectivity estimation with empty documents."""
        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        result = SelectivityAnalyzer.estimate_selectivity([], spec, schema)
        assert result == 0.0

    def test_estimate_selectivity_all_match(self):
        """Test selectivity estimation when all documents match."""
        from haystack import Document

        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        documents = [
            Document(content="doc1", meta={"category": "tech"}),
            Document(content="doc2", meta={"category": "tech"}),
        ]
        result = SelectivityAnalyzer.estimate_selectivity(documents, spec, schema)
        assert result == 1.0

    def test_estimate_selectivity_none_match(self):
        """Test selectivity estimation when no documents match."""
        from haystack import Document

        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        documents = [
            Document(content="doc1", meta={"category": "science"}),
            Document(content="doc2", meta={"category": "history"}),
        ]
        result = SelectivityAnalyzer.estimate_selectivity(documents, spec, schema)
        assert result == 0.0

    def test_estimate_selectivity_partial_match(self):
        """Test selectivity estimation with partial matches."""
        from haystack import Document

        schema = {
            "category": FilterField(
                name="category", type="string", operators=["eq"], description="Category"
            )
        }
        spec = FilterSpec(
            conditions=[FilterCondition(field="category", operator="eq", value="tech")]
        )
        documents = [
            Document(content="doc1", meta={"category": "tech"}),
            Document(content="doc2", meta={"category": "science"}),
            Document(content="doc3", meta={"category": "tech"}),
            Document(content="doc4", meta={"category": "history"}),
        ]
        result = SelectivityAnalyzer.estimate_selectivity(documents, spec, schema)
        assert result == 0.5

    def test_matches_condition_eq(self):
        """Test matching condition with eq operator."""
        from haystack import Document

        doc = Document(content="test", meta={"category": "tech"})
        condition = FilterCondition(field="category", operator="eq", value="tech")
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

        doc2 = Document(content="test", meta={"category": "science"})
        assert SelectivityAnalyzer._matches_condition(doc2, condition, {}) is False

    def test_matches_condition_ne(self):
        """Test matching condition with ne operator."""
        from haystack import Document

        doc = Document(content="test", meta={"category": "tech"})
        condition = FilterCondition(field="category", operator="ne", value="science")
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

    def test_matches_condition_gt(self):
        """Test matching condition with gt operator."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2024})
        condition = FilterCondition(field="year", operator="gt", value=2020)
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

        doc2 = Document(content="test", meta={"year": 2019})
        assert SelectivityAnalyzer._matches_condition(doc2, condition, {}) is False

    def test_matches_condition_gte(self):
        """Test matching condition with gte operator."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2020})
        condition = FilterCondition(field="year", operator="gte", value=2020)
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

    def test_matches_condition_lt(self):
        """Test matching condition with lt operator."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2019})
        condition = FilterCondition(field="year", operator="lt", value=2020)
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

    def test_matches_condition_lte(self):
        """Test matching condition with lte operator."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2020})
        condition = FilterCondition(field="year", operator="lte", value=2020)
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

    def test_matches_condition_in(self):
        """Test matching condition with in operator."""
        from haystack import Document

        doc = Document(content="test", meta={"category": "tech"})
        condition = FilterCondition(
            field="category", operator="in", value=["tech", "science"]
        )
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

        doc2 = Document(content="test", meta={"category": "history"})
        assert SelectivityAnalyzer._matches_condition(doc2, condition, {}) is False

    def test_matches_condition_contains(self):
        """Test matching condition with contains operator."""
        from haystack import Document

        doc = Document(content="test", meta={"text": "hello world"})
        condition = FilterCondition(field="text", operator="contains", value="world")
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

        doc2 = Document(content="test", meta={"text": "hello universe"})
        assert SelectivityAnalyzer._matches_condition(doc2, condition, {}) is False

    def test_matches_condition_range(self):
        """Test matching condition with range operator."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2022})
        condition = FilterCondition(field="year", operator="range", value=(2020, 2024))
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is True

        doc2 = Document(content="test", meta={"year": 2019})
        assert SelectivityAnalyzer._matches_condition(doc2, condition, {}) is False

        doc3 = Document(content="test", meta={"year": 2025})
        assert SelectivityAnalyzer._matches_condition(doc3, condition, {}) is False

    def test_matches_condition_range_invalid_value(self):
        """Test matching condition with invalid range value."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2022})
        condition = FilterCondition(field="year", operator="range", value="bad")
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is False

    def test_matches_condition_range_invalid_tuple_length(self):
        """Test matching condition with short range tuple."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2022})
        condition = FilterCondition(field="year", operator="range", value=(2020,))
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is False

    def test_matches_condition_unsupported_operator(self):
        """Test matching condition with unsupported operator."""
        from haystack import Document

        doc = Document(content="test", meta={"year": 2022})
        condition = make_invalid_condition("year", "unsupported", 2020)
        assert SelectivityAnalyzer._matches_condition(doc, condition, {}) is False

    def test_matches_all_conditions(self):
        """Test matching all conditions (AND logic)."""
        from haystack import Document

        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="eq", value="tech"),
                FilterCondition(field="year", operator="gte", value=2020),
            ]
        )
        doc = Document(content="test", meta={"category": "tech", "year": 2022})
        assert SelectivityAnalyzer._matches_all_conditions(doc, spec, {}) is True

        doc2 = Document(content="test", meta={"category": "science", "year": 2022})
        assert SelectivityAnalyzer._matches_all_conditions(doc2, spec, {}) is False

        doc3 = Document(content="test", meta={"category": "tech", "year": 2019})
        assert SelectivityAnalyzer._matches_all_conditions(doc3, spec, {}) is False
