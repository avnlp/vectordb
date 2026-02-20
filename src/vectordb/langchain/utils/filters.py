"""Document filtering utilities for LangChain pipelines."""

from typing import Any, Callable

from langchain_core.documents import Document


class DocumentFilter:
    """Helper for filtering documents by metadata."""

    @staticmethod
    def filter_by_metadata(
        documents: list[Document],
        key: str,
        value: Any,
        operator: str = "equals",
    ) -> list[Document]:
        """Filter documents by metadata key-value pair.

        Args:
            documents: List of documents to filter.
            key: Metadata key to filter on.
            value: Value to match.
            operator: Filter operator (equals, contains, startswith, endswith,
                gt, lt, gte, lte, in, not_in). Note: string operators
                (contains, startswith, endswith) are case-insensitive.

        Returns:
            Filtered list of documents.

        """

        def check_condition(doc: Document) -> bool:  # noqa: PLR0911
            if key not in doc.metadata:
                return False

            metadata_value = doc.metadata[key]

            match operator:
                case "equals":
                    return metadata_value == value
                case "contains":
                    return str(value).lower() in str(metadata_value).lower()
                case "startswith":
                    return str(metadata_value).lower().startswith(str(value).lower())
                case "endswith":
                    return str(metadata_value).lower().endswith(str(value).lower())
                case "gt":
                    return metadata_value > value
                case "lt":
                    return metadata_value < value
                case "gte":
                    return metadata_value >= value
                case "lte":
                    return metadata_value <= value
                case "in":
                    return metadata_value in value
                case "not_in":
                    return metadata_value not in value
                case _:
                    msg = f"Unknown operator: {operator}"
                    raise ValueError(msg)

        return [doc for doc in documents if check_condition(doc)]

    @staticmethod
    def filter_by_predicate(
        documents: list[Document],
        predicate: Callable[[Document], bool],
    ) -> list[Document]:
        """Filter documents using a custom predicate function.

        Args:
            documents: List of documents to filter.
            predicate: Function that takes a Document and returns bool.

        Returns:
            Filtered list of documents.
        """
        return [doc for doc in documents if predicate(doc)]

    @staticmethod
    def filter_by_metadata_json(
        documents: list[Document],
        json_path: str,
        value: Any,
        operator: str = "equals",
    ) -> list[Document]:
        """Filter documents by nested JSON path in metadata.

        Args:
            documents: List of documents to filter.
            json_path: Dot-separated path (e.g., "author.name").
            value: Value to match.
            operator: Filter operator (equals, contains, startswith, endswith,
                gt, lt, gte, lte, in, not_in). Note: string operators
                (contains, startswith, endswith) are case-insensitive.

        Returns:
            Filtered list of documents.
        """

        def get_nested_value(obj: Any, path: str) -> Any:
            parts = path.split(".")
            current = obj
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            return current

        def check_condition(doc: Document) -> bool:  # noqa: PLR0911
            nested_value = get_nested_value(doc.metadata, json_path)
            if nested_value is None:
                return False

            match operator:
                case "equals":
                    return nested_value == value
                case "contains":
                    return str(value).lower() in str(nested_value).lower()
                case "startswith":
                    return str(nested_value).lower().startswith(str(value).lower())
                case "endswith":
                    return str(nested_value).lower().endswith(str(value).lower())
                case "gt":
                    return nested_value > value
                case "lt":
                    return nested_value < value
                case "gte":
                    return nested_value >= value
                case "lte":
                    return nested_value <= value
                case "in":
                    return nested_value in value
                case "not_in":
                    return nested_value not in value
                case _:
                    msg = f"Unknown operator: {operator}"
                    raise ValueError(msg)

        return [doc for doc in documents if check_condition(doc)]

    @staticmethod
    def exclude_by_metadata(
        documents: list[Document],
        key: str,
        value: Any,
    ) -> list[Document]:
        """Exclude documents by metadata key-value pair.

        Args:
            documents: List of documents to filter.
            key: Metadata key.
            value: Value to exclude.

        Returns:
            Filtered list with matching documents removed.
        """
        return [doc for doc in documents if doc.metadata.get(key) != value]
