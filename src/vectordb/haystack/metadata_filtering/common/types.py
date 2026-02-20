"""Core data types for metadata filtering pipelines.

Provides dataclasses and types for filter specifications, timing metrics,
and query results shared across all vector database implementations.
"""

from dataclasses import dataclass
from typing import Any

from haystack import Document


__all__ = [
    "FilterField",
    "FilterCondition",
    "FilterSpec",
    "TimingMetrics",
    "FilteredQueryResult",
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
    indexed: bool = True
    nullable: bool = False


@dataclass
class FilterCondition:
    """Single filter condition.

    Attributes:
        field: Field name to filter on.
        operator: Comparison operator (eq, ne, gt, gte, lt, lte, in,
            contains, range).
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
