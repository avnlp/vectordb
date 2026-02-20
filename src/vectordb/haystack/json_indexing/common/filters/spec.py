"""Filter specification and validation for JSON indexing pipelines."""

# Supported operators across all DBs
SUPPORTED_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}


def validate_filter_operator(op: str) -> bool:
    """Validate if an operator is supported.

    Args:
        op: The operator to validate.

    Returns:
        True if operator is supported, False otherwise.
    """
    return op in SUPPORTED_OPERATORS
