"""Document filter class - SINGLE SOURCE OF TRUTH for all features."""

from typing import Any, Callable

from haystack import Document


class DocumentFilter:
    """Handles document filtering based on metadata criteria.

    Supported operators:
        - Simple equality: {"field": "value"}
        - $eq: {"field": {"$eq": "value"}}
        - $ne: {"field": {"$ne": "value"}}
        - $gt, $gte, $lt, $lte: Numeric comparisons
        - $in, $nin: List membership
        - $contains: Substring matching
    """

    OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
        "$eq": lambda a, v: a == v,
        "$ne": lambda a, v: a != v,
        "$gt": lambda a, v: a > v,
        "$gte": lambda a, v: a >= v,
        "$lt": lambda a, v: a < v,
        "$lte": lambda a, v: a <= v,
        "$in": lambda a, v: a in v,
        "$nin": lambda a, v: a not in v,
        "$contains": lambda a, v: v in str(a),
    }

    @classmethod
    def normalize(cls, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Normalize filter dict to canonical format.

        Args:
            filters: Raw filter dictionary or None.

        Returns:
            Normalized filter dictionary or None if empty.
        """
        if not filters:
            return None
        return filters

    @classmethod
    def apply(
        cls, documents: list[Document], filters: dict[str, Any] | None
    ) -> list[Document]:
        """Filter documents in Python based on metadata criteria.

        Use this as fallback when VectorDB doesn't support server-side filtering.
        For performance, prefer passing filters to VectorDB.search() when supported.

        Args:
            documents: List of documents to filter.
            filters: Filter conditions.

        Returns:
            Filtered list of documents.
        """
        if not filters:
            return documents

        return [doc for doc in documents if cls._matches_filters(doc.meta, filters)]

    @classmethod
    def _matches_filters(
        cls, meta: dict[str, Any] | None, filters: dict[str, Any]
    ) -> bool:
        """Check if document metadata matches all filter conditions."""
        if meta is None:
            return False
        for field, expected in filters.items():
            actual = meta.get(field)

            if isinstance(expected, dict):
                for op, value in expected.items():
                    if not cls._eval_operator(actual, op, value):
                        return False
            elif actual != expected:
                return False

        return True

    @classmethod
    def _eval_operator(cls, actual: Any, op: str, value: Any) -> bool:
        """Evaluate a single filter operator."""
        if actual is None:
            return op == "$ne"

        evaluator = cls.OPERATORS.get(op)
        if evaluator is None:
            return True

        try:
            return evaluator(actual, value)
        except (TypeError, ValueError):
            return False
