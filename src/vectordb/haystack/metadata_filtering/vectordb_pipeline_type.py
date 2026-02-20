"""Core utilities for metadata filtering pipelines.

Provides shared dataclasses, filter expression builders, validators, and utilities
for metadata filtering across all vector databases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from haystack import Document


__all__ = [
    "FilterField",
    "FilterCondition",
    "FilterSpec",
    "TimingMetrics",
    "FilteredQueryResult",
    "FilterExpressionBuilder",
    "MilvusFilterExpressionBuilder",
    "QdrantFilterExpressionBuilder",
    "PineconeFilterExpressionBuilder",
    "WeaviateFilterExpressionBuilder",
    "ChromaFilterExpressionBuilder",
    "parse_filter_from_config",
    "validate_filter_config",
    "SelectivityAnalyzer",
]


@dataclass
class FilterField:
    """Metadata field definition for filtering.

    Attributes:
        name: Field name in metadata.
        type: Data type (string, integer, float, boolean).
        operators: Supported filter operators (eq, ne, gt, gte, lt, lte,
            in, contains, range).
        description: Human-readable description.
        indexed: Whether field is indexed (default: True).
        nullable: Whether field can be None (default: False).
    """

    name: str
    type: str
    operators: list[str]
    description: str
    indexed: bool = True  # noqa: W505
    nullable: bool = False


@dataclass
class FilterCondition:
    """Single filter condition.

    Attributes:
        field: Field name to filter on.
        operator: Comparison operator (eq, ne, gt, gte, lt, lte, in, contains, range).
        value: Value(s) to compare against.

    Raises:
        ValueError: If operator is not valid.
    """

    field: str
    operator: str
    value: Any

    VALID_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "contains", "range"}

    def __post_init__(self) -> None:
        """Validate operator during initialization."""
        if self.operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator: {self.operator}. "
                f"Must be one of {self.VALID_OPERATORS}"
            )


@dataclass
class FilterSpec:
    """Complete filter specification (AND of conditions).

    Attributes:
        conditions: List of FilterCondition objects to AND together.
    """

    conditions: list[FilterCondition]


@dataclass
class TimingMetrics:
    """Timing metrics for pre-filtering and vector search.

    Attributes:
        pre_filter_ms: Time to apply pre-filter (milliseconds).
        vector_search_ms: Time to run vector search (milliseconds).
        total_ms: Total time (pre-filter + vector search).
        num_candidates: Number of documents matching filter.
        num_total_docs: Total documents in corpus.
    """

    pre_filter_ms: float
    vector_search_ms: float
    total_ms: float
    num_candidates: int
    num_total_docs: int

    @property
    def selectivity(self) -> float:
        """Compute selectivity as fraction of candidates.

        Returns:
            Fraction of documents matching filter (0.0 to 1.0).
            Returns 0.0 if num_total_docs is 0.
        """
        if self.num_total_docs == 0:
            return 0.0
        return self.num_candidates / self.num_total_docs


@dataclass
class FilteredQueryResult:
    """Result from a filtered query.

    Attributes:
        document: Retrieved document.
        relevance_score: Vector similarity score.
        rank: Rank in results (1-indexed).
        filter_matched: Whether document matched filter.
        timing: Optional timing metrics.
    """

    document: Document
    relevance_score: float
    rank: int
    filter_matched: bool
    timing: TimingMetrics | None = None


class FilterExpressionBuilder(ABC):
    """Base class for building DB-specific filter expressions.

    Converts universal FilterSpec to database-native syntax.
    Each vector database has unique filter expression formats.
    """

    def __init__(self, schema: dict[str, FilterField]) -> None:
        """Initialize builder with metadata schema.

        Args:
            schema: Dict mapping field names to FilterField definitions.
        """
        self.schema = schema

    @abstractmethod
    def build(self, spec: FilterSpec) -> Any:
        """Build DB-specific filter expression from FilterSpec.

        Args:
            spec: Universal filter specification.

        Returns:
            DB-native filter expression (string, dict, object, etc.).
        """

    def validate_condition(self, condition: FilterCondition) -> None:
        """Validate condition against schema.

        Args:
            condition: FilterCondition to validate.

        Raises:
            ValueError: If field not in schema or operator unsupported for field.
        """
        if condition.field not in self.schema:
            raise ValueError(f"Field not in schema: {condition.field}")

        field_def = self.schema[condition.field]
        if condition.operator not in field_def.operators:
            raise ValueError(
                f"Operator {condition.operator} not supported for field {condition.field}. "
                f"Supported: {field_def.operators}"
            )


class MilvusFilterExpressionBuilder(FilterExpressionBuilder):
    """Builds Milvus string filter expressions.

    Milvus uses string expressions like:
      metadata["field"] == value && metadata["other"] >= 10
    """

    def build(self, spec: FilterSpec) -> str:
        """Build Milvus filter expression string.

        Args:
            spec: Filter specification.

        Returns:
            Milvus-compatible filter string.
        """
        if not spec.conditions:
            return ""

        expressions = []
        for condition in spec.conditions:
            self.validate_condition(condition)
            expr = self._build_condition(condition)
            expressions.append(expr)

        return " && ".join(expressions)

    def _build_condition(self, condition: FilterCondition) -> str:  # noqa: PLR0911
        """Build single Milvus condition expression.

        Args:
            condition: Single filter condition.

        Returns:
            Milvus expression string.

        Raises:
            ValueError: For unsupported operators.
        """
        field_expr = f'metadata["{condition.field}"]'

        if condition.operator == "eq":
            return f"{field_expr} == {self._format_value(condition.value)}"
        if condition.operator == "ne":
            return f"{field_expr} != {self._format_value(condition.value)}"
        if condition.operator == "gt":
            return f"{field_expr} > {self._format_value(condition.value)}"
        if condition.operator == "gte":
            return f"{field_expr} >= {self._format_value(condition.value)}"
        if condition.operator == "lt":
            return f"{field_expr} < {self._format_value(condition.value)}"
        if condition.operator == "lte":
            return f"{field_expr} <= {self._format_value(condition.value)}"
        if condition.operator == "in":
            if not isinstance(condition.value, list):
                raise ValueError(
                    f"'in' operator requires list, got {type(condition.value)}"
                )
            formatted_values = [self._format_value(v) for v in condition.value]
            return f"{field_expr} in [{', '.join(formatted_values)}]"
        if condition.operator == "contains":
            return f'{field_expr} like "%{condition.value}%"'
        if condition.operator == "range":
            if (
                not isinstance(condition.value, (list, tuple))
                or len(condition.value) != 2
            ):
                raise ValueError("'range' operator requires (min, max) tuple")
            min_val, max_val = condition.value
            return (
                f"{field_expr} >= {self._format_value(min_val)} && "
                f"{field_expr} <= {self._format_value(max_val)}"
            )

        raise ValueError(f"Unsupported operator: {condition.operator}")

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value for Milvus expression.

        Args:
            value: Value to format.

        Returns:
            Formatted value string.
        """
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


class QdrantFilterExpressionBuilder(FilterExpressionBuilder):
    """Builds Qdrant Filter objects.

    Qdrant uses structured Filter objects with must/should/must_not conditions.
    """

    def build(self, spec: FilterSpec) -> Any:
        """Build Qdrant Filter with must conditions.

        Args:
            spec: Filter specification.

        Returns:
            qdrant_client.models.Filter object.
        """
        from qdrant_client import models

        must_conditions = []
        for condition in spec.conditions:
            self.validate_condition(condition)
            cond = self._build_condition(condition)
            must_conditions.append(cond)

        return models.Filter(must=must_conditions)

    def _build_condition(self, condition: FilterCondition) -> Any:
        """Build single Qdrant FieldCondition.

        Args:
            condition: Single filter condition.

        Returns:
            qdrant_client.models.FieldCondition object.

        Raises:
            ValueError: For unsupported operators or invalid values.
        """
        from qdrant_client import models

        if condition.operator == "eq":
            return models.FieldCondition(
                key=condition.field, match=models.MatchValue(value=condition.value)
            )
        if condition.operator == "ne":
            return models.FieldCondition(
                key=condition.field, match=models.MatchValue(value=condition.value)
            )
        if condition.operator in ["gt", "gte", "lt", "lte"]:
            op_map = {"gt": "gt", "gte": "gte", "lt": "lt", "lte": "lte"}
            range_kwargs = {op_map[condition.operator]: condition.value}
            return models.FieldCondition(
                key=condition.field, range=models.Range(**range_kwargs)
            )
        if condition.operator == "in":
            return models.FieldCondition(
                key=condition.field, match=models.MatchAny(any=condition.value)
            )
        if condition.operator == "contains":
            return models.FieldCondition(
                key=condition.field, match=models.MatchText(text=condition.value)
            )
        if condition.operator == "range":
            if (
                not isinstance(condition.value, (list, tuple))
                or len(condition.value) != 2
            ):
                raise ValueError("'range' requires (min, max) tuple")
            min_val, max_val = condition.value
            return models.FieldCondition(
                key=condition.field, range=models.Range(gte=min_val, lte=max_val)
            )

        raise ValueError(f"Unsupported operator: {condition.operator}")


class PineconeFilterExpressionBuilder(FilterExpressionBuilder):
    """Builds Pinecone MongoDB-style filter dicts.

    Pinecone uses MongoDB query syntax: {"field": {"$op": value}, ...}
    """

    def build(self, spec: FilterSpec) -> dict[str, Any]:
        """Build Pinecone filter dict.

        Args:
            spec: Filter specification.

        Returns:
            MongoDB-style filter dictionary.
        """
        if not spec.conditions:
            return {}

        conditions_dicts = []
        for condition in spec.conditions:
            self.validate_condition(condition)
            cond_dict = self._build_condition(condition)
            conditions_dicts.append(cond_dict)

        if len(conditions_dicts) == 1:
            return conditions_dicts[0]
        return {"$and": conditions_dicts}

    def _build_condition(self, condition: FilterCondition) -> dict[str, Any]:  # noqa: PLR0911
        """Build single Pinecone condition as dict.

        Args:
            condition: Single filter condition.

        Returns:
            Dict representing condition.

        Raises:
            ValueError: For unsupported operators or invalid values.
        """
        if condition.operator == "eq":
            return {condition.field: {"$eq": condition.value}}
        if condition.operator == "ne":
            return {condition.field: {"$ne": condition.value}}
        if condition.operator == "gt":
            return {condition.field: {"$gt": condition.value}}
        if condition.operator == "gte":
            return {condition.field: {"$gte": condition.value}}
        if condition.operator == "lt":
            return {condition.field: {"$lt": condition.value}}
        if condition.operator == "lte":
            return {condition.field: {"$lte": condition.value}}
        if condition.operator == "in":
            return {condition.field: {"$in": condition.value}}
        if condition.operator == "contains":
            return {condition.field: {"$exists": True}}
        if condition.operator == "range":
            if (
                not isinstance(condition.value, (list, tuple))
                or len(condition.value) != 2
            ):
                raise ValueError("'range' requires (min, max) tuple")
            min_val, max_val = condition.value
            return {
                "$and": [
                    {condition.field: {"$gte": min_val}},
                    {condition.field: {"$lte": max_val}},
                ]
            }

        raise ValueError(f"Unsupported operator: {condition.operator}")


class WeaviateFilterExpressionBuilder(FilterExpressionBuilder):
    """Builds Weaviate GraphQL-like where clause dicts.

    Weaviate uses structured where clauses:
      {"path": ["field"], "operator": "Equal", "valueString": "..."}
    """

    def build(self, spec: FilterSpec) -> dict[str, Any]:
        """Build Weaviate where clause dict.

        Args:
            spec: Filter specification.

        Returns:
            Weaviate where clause dictionary.
        """
        if not spec.conditions:
            return {}

        operands = []
        for condition in spec.conditions:
            self.validate_condition(condition)
            operand = self._build_condition(condition)
            operands.append(operand)

        if len(operands) == 1:
            return operands[0]
        return {"operator": "And", "operands": operands}

    def _build_condition(self, condition: FilterCondition) -> dict[str, Any]:  # noqa: PLR0911
        """Build single Weaviate operand.

        Args:
            condition: Single filter condition.

        Returns:
            Dict representing Weaviate condition.

        Raises:
            ValueError: For unsupported operators or invalid values.
        """
        field_path = [condition.field]

        if condition.operator == "eq":
            value_key = (
                "valueString" if isinstance(condition.value, str) else "valueNumber"
            )
            return {"path": field_path, "operator": "Equal", value_key: condition.value}
        if condition.operator == "ne":
            value_key = (
                "valueString" if isinstance(condition.value, str) else "valueNumber"
            )
            return {
                "path": field_path,
                "operator": "NotEqual",
                value_key: condition.value,
            }
        if condition.operator == "gt":
            return {
                "path": field_path,
                "operator": "GreaterThan",
                "valueNumber": condition.value,
            }
        if condition.operator == "gte":
            return {
                "path": field_path,
                "operator": "GreaterThanEqual",
                "valueNumber": condition.value,
            }
        if condition.operator == "lt":
            return {
                "path": field_path,
                "operator": "LessThan",
                "valueNumber": condition.value,
            }
        if condition.operator == "lte":
            return {
                "path": field_path,
                "operator": "LessThanEqual",
                "valueNumber": condition.value,
            }
        if condition.operator == "in":
            value_key = (
                "valueStringArray"
                if all(isinstance(v, str) for v in condition.value)
                else "valueNumberArray"
            )
            return {
                "path": field_path,
                "operator": "ContainsAny",
                value_key: condition.value,
            }
        if condition.operator == "contains":
            return {
                "path": field_path,
                "operator": "Like",
                "valueString": f"%{condition.value}%",
            }
        if condition.operator == "range":
            if (
                not isinstance(condition.value, (list, tuple))
                or len(condition.value) != 2
            ):
                raise ValueError("'range' requires (min, max) tuple")
            min_val, max_val = condition.value
            return {
                "operator": "And",
                "operands": [
                    {
                        "path": field_path,
                        "operator": "GreaterThanEqual",
                        "valueNumber": min_val,
                    },
                    {
                        "path": field_path,
                        "operator": "LessThanEqual",
                        "valueNumber": max_val,
                    },
                ],
            }

        raise ValueError(f"Unsupported operator: {condition.operator}")


class ChromaFilterExpressionBuilder(FilterExpressionBuilder):
    """Builds Chroma MongoDB-style filter dicts.

    Chroma uses MongoDB query syntax (same as Pinecone):
      {"field": value} or {"field": {"$op": value}}
    """

    def build(self, spec: FilterSpec) -> dict[str, Any]:
        """Build Chroma filter dict.

        Args:
            spec: Filter specification.

        Returns:
            MongoDB-style filter dictionary.
        """
        if not spec.conditions:
            return {}

        conditions_dicts = []
        for condition in spec.conditions:
            self.validate_condition(condition)
            cond_dict = self._build_condition(condition)
            conditions_dicts.append(cond_dict)

        if len(conditions_dicts) == 1:
            return conditions_dicts[0]
        return {"$and": conditions_dicts}

    def _build_condition(self, condition: FilterCondition) -> dict[str, Any]:  # noqa: PLR0911
        """Build single Chroma condition as dict.

        Args:
            condition: Single filter condition.

        Returns:
            Dict representing condition.

        Raises:
            ValueError: For unsupported operators or invalid values.
        """
        if condition.operator == "eq":
            return {condition.field: condition.value}
        if condition.operator == "ne":
            return {condition.field: {"$ne": condition.value}}
        if condition.operator == "gt":
            return {condition.field: {"$gt": condition.value}}
        if condition.operator == "gte":
            return {condition.field: {"$gte": condition.value}}
        if condition.operator == "lt":
            return {condition.field: {"$lt": condition.value}}
        if condition.operator == "lte":
            return {condition.field: {"$lte": condition.value}}
        if condition.operator == "in":
            return {condition.field: {"$in": condition.value}}
        if condition.operator == "contains":
            return {condition.field: {"$exists": True}}
        if condition.operator == "range":
            if (
                not isinstance(condition.value, (list, tuple))
                or len(condition.value) != 2
            ):
                raise ValueError("'range' requires (min, max) tuple")
            min_val, max_val = condition.value
            return {
                "$and": [
                    {condition.field: {"$gte": min_val}},
                    {condition.field: {"$lte": max_val}},
                ]
            }

        raise ValueError(f"Unsupported operator: {condition.operator}")


def parse_filter_from_config(config: dict[str, Any]) -> FilterSpec:
    """Parse filter specification from config dictionary.

    Extracts test_filters from metadata_filtering section and builds FilterSpec.

    Args:
        config: Configuration dictionary with metadata_filtering section.

    Returns:
        FilterSpec object with parsed conditions.

    Raises:
        ValueError: If test_filters not found in config.
    """
    metadata_filtering = config.get("metadata_filtering", {})
    test_filters = metadata_filtering.get("test_filters")

    if not test_filters:
        raise ValueError(
            "No test_filters found in metadata_filtering section of config"
        )

    if not isinstance(test_filters, list) or len(test_filters) == 0:
        raise ValueError("test_filters must be non-empty list")

    # Use first filter spec
    filter_spec = test_filters[0]
    conditions = []

    for cond_dict in filter_spec.get("conditions", []):
        condition = FilterCondition(
            field=cond_dict["field"],
            operator=cond_dict["operator"],
            value=cond_dict["value"],
        )
        conditions.append(condition)

    return FilterSpec(conditions)


def validate_filter_config(
    config: dict[str, Any], schema: dict[str, FilterField]
) -> tuple[bool, list[str]]:
    """Validate filter configuration against schema.

    Checks:
    - test_filters section exists
    - All referenced fields are in schema
    - All operators are supported for their fields

    Args:
        config: Configuration dictionary to validate.
        schema: FilterField schema for validation.

    Returns:
        Tuple of (is_valid: bool, errors: list[str]).
        If valid, errors list is empty.
    """
    errors = []

    metadata_filtering = config.get("metadata_filtering", {})
    test_filters = metadata_filtering.get("test_filters")

    if not test_filters:
        return False, ["No test_filters found in metadata_filtering section"]

    if not isinstance(test_filters, list) or len(test_filters) == 0:
        return False, ["test_filters must be non-empty list"]

    for filter_spec in test_filters:
        filter_name = filter_spec.get("name", "unnamed")
        conditions = filter_spec.get("conditions", [])

        if not conditions:
            errors.append(f"Filter '{filter_name}' has no conditions")
            continue

        for cond_dict in conditions:
            field_name = cond_dict.get("field")
            operator = cond_dict.get("operator")

            if field_name not in schema:
                errors.append(f"Filter '{filter_name}': Unknown field '{field_name}'")
                continue

            field_def = schema[field_name]
            if operator not in field_def.operators:
                errors.append(
                    f"Filter '{filter_name}': Operator '{operator}' "
                    f"not supported for field '{field_name}'"
                )

    return len(errors) == 0, errors


class SelectivityAnalyzer:
    """Utility class for estimating filter selectivity on a corpus.

    Selectivity = number of matching documents / total documents.
    Used to estimate filter effectiveness.
    """

    @staticmethod
    def estimate_selectivity(
        documents: list[Document], spec: FilterSpec, schema: dict[str, FilterField]
    ) -> float:
        """Estimate filter selectivity on document corpus.

        Evaluates filter against actual documents and computes:
        selectivity = matching_documents / total_documents

        Args:
            documents: Document corpus to evaluate against.
            spec: Filter specification.
            schema: Metadata field schema.

        Returns:
            Selectivity as fraction (0.0 to 1.0).
        """
        if not documents:
            return 0.0

        matching_count = 0

        for doc in documents:
            if SelectivityAnalyzer._matches_all_conditions(doc, spec, schema):
                matching_count += 1

        return matching_count / len(documents)

    @staticmethod
    def _matches_all_conditions(
        doc: Document, spec: FilterSpec, schema: dict[str, FilterField]
    ) -> bool:
        """Check if document matches all conditions in spec.

        Args:
            doc: Document to check.
            spec: Filter specification (AND of conditions).
            schema: Metadata field schema.

        Returns:
            True if document matches all conditions, False otherwise.
        """
        for condition in spec.conditions:
            if not SelectivityAnalyzer._matches_condition(doc, condition, schema):
                return False
        return True

    @staticmethod
    def _matches_condition(  # noqa: PLR0911
        doc: Document, condition: FilterCondition, schema: dict[str, FilterField]
    ) -> bool:
        """Check if document matches single condition.

        Args:
            doc: Document to check.
            condition: Single filter condition.
            schema: Metadata field schema.

        Returns:
            True if document matches condition, False otherwise.
        """
        field_value = doc.meta.get(condition.field)

        if condition.operator == "eq":
            return field_value == condition.value
        if condition.operator == "ne":
            return field_value != condition.value
        if condition.operator == "gt":
            return field_value is not None and field_value > condition.value
        if condition.operator == "gte":
            return field_value is not None and field_value >= condition.value
        if condition.operator == "lt":
            return field_value is not None and field_value < condition.value
        if condition.operator == "lte":
            return field_value is not None and field_value <= condition.value
        if condition.operator == "in":
            return (
                field_value in condition.value
                if isinstance(condition.value, list)
                else False
            )
        if condition.operator == "contains":
            return (
                isinstance(field_value, str) and condition.value in field_value
                if field_value is not None
                else False
            )
        if condition.operator == "range":
            if (
                not isinstance(condition.value, (list, tuple))
                or len(condition.value) != 2
            ):
                return False
            min_val, max_val = condition.value
            return field_value is not None and min_val <= field_value <= max_val

        return False
