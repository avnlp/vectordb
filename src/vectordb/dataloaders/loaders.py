"""Dataset loaders and registry for RAG evaluation.

This module provides a registry pattern for loading common RAG evaluation datasets
from HuggingFace. Each dataloader standardizes the dataset format to enable
evaluation across different retrieval pipelines.

Supported Datasets:
    - TriviaQA: Open-domain QA with evidence documents (HuggingFace: trivia_qa)
        Reading comprehension dataset with web evidence for trivia questions
    - ARC: AI2 Reasoning Challenge science QA (HuggingFace: ai2_arc)
        Multiple-choice science questions requiring reasoning
    - PopQA: Entity-centric factoid QA (HuggingFace: akariasai/PopQA)
        Factoid questions about popular entities
    - FactScore: Fact verification with atomic facts (HuggingFace: dskar/FActScore)
        Wikipedia-based fact verification with decomposed facts
    - Earnings Calls: Financial Q&A from EDGAR (HuggingFace: lamini/earnings-calls-qa)
        Earnings call transcripts with financial Q&A pairs

Standardized Output Format:
    Each dataloader returns a list of dicts with keys:
    - "text": The document content for indexing (str)
    - "metadata": Dict containing question, answer, and dataset-specific fields

    This format decouples data loading from framework-specific Document types,
    enabling conversion to Haystack, LangChain, or other formats as needed.

Architecture:
    The module uses a registry pattern (DatasetRegistry) to provide a unified
    interface for loading any supported dataset by type name. Individual
    dataloader classes handle dataset-specific loading logic.

    DataloaderProtocol defines the interface that all dataloaders implement,
    ensuring consistency across different dataset implementations.

Usage:
    >>> from vectordb.dataloaders import DatasetRegistry
    >>> data = DatasetRegistry.load("triviaqa", split="test", limit=100)
    >>> # Returns [{"text": "...", "metadata": {"question": "...", "answer": "..."}}]

    >>> # Register custom dataloader
    >>> from vectordb.dataloaders import DatasetRegistry, DataloaderProtocol
    >>> DatasetRegistry.register("custom", MyCustomDataloader)

Integration Points:
    - DocumentConverter: Converts standardized output to framework Documents
    - EvaluationExtractor: Extracts query-answer pairs for evaluation
    - Framework-specific dataloaders: Extend base loaders with framework integration
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from vectordb.dataloaders.arc import ARCDataloader
from vectordb.dataloaders.earnings_calls import EarningsCallDataloader
from vectordb.dataloaders.factscore import FactScoreDataloader
from vectordb.dataloaders.popqa import PopQADataloader
from vectordb.dataloaders.triviaqa import TriviaQADataloader


__all__ = [
    "DataloaderProtocol",
    "DatasetRegistry",
    "DatasetType",
    "TriviaQADataloader",
    "ARCDataloader",
    "PopQADataloader",
    "FactScoreDataloader",
    "EarningsCallDataloader",
]


class DataloaderProtocol(Protocol):
    """Interface that all base dataloaders implement.

    This protocol defines the contract for dataset loaders, ensuring
    consistent method signatures across different dataset implementations.

    Implementers must provide a load() method that returns standardized
    dict format suitable for conversion to framework-specific Documents.
    """

    def load(self) -> list[dict[str, Any]]:
        """Load dataset and return standardized format.

        Returns:
            List of dicts with "text" and "metadata" keys
        """
        ...


# Type alias for supported dataset types
# Used for type checking and IDE autocompletion
DatasetType = Literal["triviaqa", "arc", "popqa", "factscore", "earnings_calls"]


class DatasetRegistry:
    """Registry for dataset loaders.

    Provides a centralized registry for dataset loaders, enabling
    configuration-driven data loading by dataset type name.

    The registry pattern allows:
    1. Unified interface for loading any supported dataset
    2. Runtime registration of custom dataloaders
    3. Discovery of available dataset types

    Class Attributes:
        _loaders: Internal mapping of dataset type names to loader classes
    """

    # Internal registry mapping dataset type names to loader classes
    _loaders: dict[str, type[DataloaderProtocol]] = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "earnings_calls": EarningsCallDataloader,
    }

    @classmethod
    def load(
        cls,
        dataset_type: DatasetType,
        dataset_name: str | None = None,
        split: str = "test",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load a dataset by type.

        Factory method that instantiates the appropriate dataloader
        based on the dataset type and loads the data with specified
        parameters.

        Args:
            dataset_type: One of "triviaqa", "arc", "popqa", "factscore",
                "earnings_calls"
            dataset_name: HuggingFace dataset ID (optional, uses defaults
                if not provided)
            split: Dataset split to load (default: "test")
            limit: Max number of items to load (useful for sampling/testing)

        Returns:
            List of standardized dicts: {"text": str, "metadata": dict}

        Raises:
            ValueError: If dataset_type is not in the registry

        Example:
            >>> data = DatasetRegistry.load("triviaqa", split="validation", limit=50)
            >>> len(data)
            50
        """
        key = dataset_type.lower()

        if key not in cls._loaders:
            supported = list(cls._loaders.keys())
            raise ValueError(
                f"Unknown dataset: {dataset_type!r}. Supported: {supported}"
            )

        loader_cls = cls._loaders[key]

        # Build kwargs dynamically to only pass provided parameters
        kwargs: dict[str, Any] = {"split": split}
        if dataset_name is not None:
            kwargs["dataset_name"] = dataset_name
        if limit is not None:
            kwargs["limit"] = limit

        return loader_cls(**kwargs).load()

    @classmethod
    def register(cls, name: str, loader_cls: type[DataloaderProtocol]) -> None:
        """Register a new loader class.

        Enables runtime extension of supported datasets. Custom dataloaders
        must implement the DataloaderProtocol.

        Args:
            name: Dataset type name for registry lookup (case-insensitive)
            loader_cls: Dataloader class implementing DataloaderProtocol

        Example:
            >>> from vectordb.dataloaders import DatasetRegistry, DataloaderProtocol
            >>> class MyDataloader:
            ...     def load(self):
            ...         return [{"text": "...", "metadata": {}}]
            >>> DatasetRegistry.register("mydataset", MyDataloader)
        """
        cls._loaders[name.lower()] = loader_cls

    @classmethod
    def supported_datasets(cls) -> list[str]:
        """Return list of supported dataset types.

        Returns:
            List of registered dataset type names

        Example:
            >>> DatasetRegistry.supported_datasets()
            ['triviaqa', 'arc', 'popqa', 'factscore', 'earnings_calls']
        """
        return list(cls._loaders.keys())
