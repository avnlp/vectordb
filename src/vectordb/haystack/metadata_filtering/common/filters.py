"""Filter parsing and conversion utilities for metadata filtering pipelines.

Converts configuration-based filters to canonical dict format for VectorDB wrappers.
"""

from typing import Any

from vectordb.haystack.metadata_filtering.common.types import (
    FilterCondition,
    FilterSpec,
)


__all__ = ["parse_filter_from_config", "filter_spec_to_canonical_dict"]


def parse_filter_from_config(config: dict[str, Any]) -> FilterSpec:
    """Parse filter specification from configuration.

    Expected config structure:
    metadata_filtering:
      test_filters:
        - name: filter_name
          description: "..."
          conditions:
            - field: field_name
              operator: eq|ne|gt|gte|lt|lte|in|contains|range
              value: value

    Args:
        config: Configuration dictionary.

    Returns:
        FilterSpec with conditions to apply.

    Raises:
        ValueError: If filter configuration is invalid.
    """
    metadata_filtering = config.get("metadata_filtering", {})
    test_filters = metadata_filtering.get("test_filters", [])

    if not test_filters:
        return FilterSpec(conditions=[])

    # Use first test filter (can be extended to support multiple)
    filter_def = test_filters[0]
    conditions_data = filter_def.get("conditions", [])

    conditions = []
    for cond_data in conditions_data:
        condition = FilterCondition(
            field=cond_data["field"],
            operator=cond_data.get("operator", "eq"),
            value=cond_data["value"],
        )
        conditions.append(condition)

    return FilterSpec(conditions=conditions)


def filter_spec_to_canonical_dict(spec: FilterSpec) -> dict[str, Any]:
    """Convert FilterSpec to canonical dict format for VectorDB wrappers.

    Converts from FilterSpec (list of conditions) to a nested dict that
    can be passed to VectorDB wrappers' search methods.

    Args:
        spec: Filter specification to convert.

    Returns:
        Canonical filter dict suitable for VectorDB wrappers.
    """
    if not spec.conditions:
        return {}

    # For single condition, return simple dict
    if len(spec.conditions) == 1:
        cond = spec.conditions[0]
        return {
            "field": cond.field,
            "operator": cond.operator,
            "value": cond.value,
        }

    # For multiple conditions, create AND structure
    return {
        "operator": "and",
        "conditions": [
            {
                "field": cond.field,
                "operator": cond.operator,
                "value": cond.value,
            }
            for cond in spec.conditions
        ],
    }
