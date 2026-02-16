"""Dataloader factory for creating dataloaders from configuration.

This module provides a unified interface for creating dataloaders across
different datasets and frameworks (Haystack/LangChain). It handles the
complexity of instantiating the correct dataloader type with appropriate
parameters, including LLM generators for datasets requiring answer summarization.

Design Rationale:
    The factory pattern centralizes dataloader creation logic, enabling
    configuration-driven pipeline construction. This supports YAML-based
    pipeline definitions where the dataloader type is specified as a string.

Generator Integration:
    Some datasets (TriviaQA, PopQA, FactScore) require LLM-based answer
    summarization to consolidate multiple answer aliases into a single
    canonical answer. The factory automatically creates the appropriate
    generator (Haystack OpenAIChatGenerator or LangChain ChatGroq) based
    on the target framework.

Configuration Schema:
    The config dict expects the following structure:
    {
        "dataloader": {
            "type": "triviaqa" | "arc" | "popqa" | "factscore" | "earnings_calls",
            "dataset_name": "optional/huggingface-id",
            "split": "test" | "train" | "validation"
        },
        "generator": {  # Required for triviaqa, popqa, factscore
            "model": "llama-3.3-70b-versatile",
            "api_key": "optional-api-key",
            "kwargs": {"temperature": 0.5, "max_tokens": 2048}
        }
    }

Framework Support:
    - Haystack: Creates Haystack-compatible dataloaders with OpenAIChatGenerator
    - LangChain: Creates LangChain-compatible dataloaders with ChatGroq

Environment Dependencies:
    Requires GROQ_API_KEY environment variable or api_key in config for
    datasets needing answer summarization.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Protocol

from haystack import Document


class DataloaderProtocol(Protocol):
    """Protocol defining the interface for framework-specific dataloaders.

    All framework-specific dataloaders (Haystack and LangChain variants)
    must implement this protocol to ensure consistent behavior across
    different pipeline implementations.

    The protocol requires two methods:
    - load_data(): Returns standardized dict format for raw data access
    - get_documents(): Returns framework-specific Document objects
    """

    def load_data(self) -> list[dict[str, Any]]:
        """Load and process the dataset.

        Returns:
            List of standardized dicts with "text" and "metadata" keys
        """
        ...

    def get_documents(self) -> list[Document]:
        """Return Haystack Documents from the corpus.

        Returns:
            List of framework-specific Document objects ready for indexing
        """
        ...


def create_haystack_generator(config: dict[str, Any]) -> Any:
    """Create a Haystack LLM generator using OpenAIChatGenerator with Groq API.

    Configures an OpenAIChatGenerator to use Groq's API endpoint for
    answer summarization. Applies sensible defaults for temperature
    and max_tokens if not specified.

    Args:
        config: Configuration dictionary containing generator settings.
            Expected structure: {"generator": {"model": "...",
            "api_key": "...", "kwargs": {...}}}

    Returns:
        OpenAIChatGenerator configured for Groq API

    Raises:
        ValueError: If GROQ_API_KEY is not available in config or environment

    Environment Variables:
        GROQ_API_KEY: API key for Groq (used if not in config)
    """
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    generator_config = config.get("generator", {})
    model = generator_config.get("model", "llama-3.3-70b-versatile")
    api_key = generator_config.get("api_key") or os.environ.get("GROQ_API_KEY")

    if not api_key:
        msg = (
            "GROQ_API_KEY required for this dataloader. Set it as environment variable."
        )
        raise ValueError(msg)

    # Apply sensible defaults for generation parameters
    # Temperature 0.5 balances creativity with consistency for answer summarization
    # Max tokens 2048 provides ample space for detailed answers
    generation_kwargs = generator_config.get("kwargs", {})
    if "temperature" not in generation_kwargs:
        generation_kwargs["temperature"] = 0.5
    if "max_tokens" not in generation_kwargs:
        generation_kwargs["max_tokens"] = 2048

    return OpenAIChatGenerator(
        api_key=Secret.from_token(api_key),
        model=model,
        api_base_url="https://api.groq.com/openai/v1",
        generation_kwargs=generation_kwargs,
    )


def create_langchain_generator(config: dict[str, Any]) -> Any:
    """Create a LangChain LLM generator using ChatGroq.

    Configures a ChatGroq instance for answer summarization. Applies
    sensible defaults for temperature and max_tokens if not specified.

    Args:
        config: Configuration dictionary containing generator settings.
            Expected structure: {"generator": {"model": "...",
            "api_key": "...", "kwargs": {...}}}

    Returns:
        ChatGroq instance from langchain-groq

    Raises:
        ValueError: If GROQ_API_KEY is not available in config or environment

    Environment Variables:
        GROQ_API_KEY: API key for Groq (used if not in config)
    """
    from langchain_groq import ChatGroq

    generator_config = config.get("generator", {})
    model = generator_config.get("model", "llama-3.3-70b-versatile")
    api_key = generator_config.get("api_key") or os.environ.get("GROQ_API_KEY")

    if not api_key:
        msg = (
            "GROQ_API_KEY required for this dataloader. Set it as environment variable."
        )
        raise ValueError(msg)

    # Apply sensible defaults for generation parameters
    kwargs = generator_config.get("kwargs", {})
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0.5
    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 2048

    return ChatGroq(model=model, api_key=api_key, **kwargs)


def create_generator(
    config: dict[str, Any],
    framework: Literal["haystack", "langchain"] = "haystack",
) -> Any:
    """Create an LLM generator for dataloaders that need answer summarization.

    Factory function that delegates to framework-specific generator creation.
    Some datasets (TriviaQA, PopQA, FactScore) use LLM summarization to
    consolidate multiple answer aliases into canonical answers.

    Args:
        config: Configuration dictionary containing generator settings
        framework: Target framework - "haystack" or "langchain"

    Returns:
        Generator instance appropriate for the framework

    Raises:
        ValueError: If GROQ_API_KEY is not available
    """
    if framework == "langchain":
        return create_langchain_generator(config)
    return create_haystack_generator(config)


def create_dataloader(config: dict[str, Any]) -> DataloaderProtocol:
    """Create a dataloader based on configuration.

    Factory function that instantiates the appropriate dataloader class
    based on the type specified in the configuration. Automatically
    handles generator injection for datasets requiring answer summarization.

    Supports the following dataset types:
    - triviaqa: TriviaQA open-domain QA (requires generator for answer summarization)
    - arc: ARC science QA (no generator needed)
    - popqa: PopQA factoid QA (requires generator for answer summarization)
    - factscore: FactScore fact verification (requires generator for
        answer summarization)
    - earnings_calls: EDGAR/Earnings Call financial documents
        (no generator needed)

    Args:
        config: Configuration dictionary containing dataloader settings.
            Expected structure: {"dataloader": {"type": "...",
            "dataset_name": "...", "split": "..."}}

    Returns:
        An initialized dataloader instance implementing DataloaderProtocol

    Raises:
        ValueError: If the dataloader type is not supported or if a required
            generator cannot be created (missing API key)
    """
    from vectordb.dataloaders.haystack import (
        ARCDataloader,
        EarningsCallDataloader,
        FactScoreDataloader,
        PopQADataloader,
        TriviaQADataloader,
    )

    dataloader_config = config.get("dataloader", {})
    dataloader_type = dataloader_config.get("type", "").lower()
    dataset_name = dataloader_config.get("dataset_name")
    split = dataloader_config.get("split", "test")

    # Datasets requiring LLM-based answer summarization
    # These have multiple answer aliases that need consolidation
    needs_generator = {"triviaqa", "popqa", "factscore"}

    generator = None
    if dataloader_type in needs_generator:
        generator = create_generator(config)

    if dataloader_type == "triviaqa":
        return TriviaQADataloader(
            answer_summary_generator=generator,
            dataset_name=dataset_name or "trivia_qa",
            split=split,
        )

    if dataloader_type == "arc":
        return ARCDataloader(
            dataset_name=dataset_name or "ai2_arc",
            split=split,
        )

    if dataloader_type == "popqa":
        return PopQADataloader(
            answer_summary_generator=generator,
            dataset_name=dataset_name or "akariasai/PopQA",
            split=split,
        )

    if dataloader_type == "factscore":
        return FactScoreDataloader(
            answer_summary_generator=generator,
            dataset_name=dataset_name or "dskar/FActScore",
            split=split,
        )

    if dataloader_type == "earnings_calls":
        return EarningsCallDataloader(
            dataset_name=dataset_name or "lamini/earnings-calls-qa",
            split=split,
        )

    supported = ["triviaqa", "arc", "popqa", "factscore", "earnings_calls"]
    msg = f"Unsupported dataloader type: '{dataloader_type}'. Supported: {supported}"
    raise ValueError(msg)


def extract_queries_and_ground_truth(
    config: dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Extract queries and ground truth from a dataset for evaluation.

    Creates query-document pairs where each query maps to its
    relevant document IDs for evaluation. Leverages EvaluationExtractor
    for consistent dataset handling.

    Args:
        config: Configuration dictionary with dataloader settings
        limit: Optional limit on number of queries to extract

    Returns:
        List of dicts with structure:
        {
            "query": str,  # The question text
            "answer": str,  # Ground truth answer
            "relevant_doc_ids": list[str]  # IDs of relevant documents
        }

    Note:
        Earnings calls dataset returns empty relevant_doc_ids as it lacks
        explicit document ID mappings in its structure.
    """
    from vectordb.dataloaders.evaluation import EvaluationExtractor

    dataloader_config = config.get("dataloader", {})
    dataset_type = dataloader_config.get("type", "").lower()
    dataset_name = dataloader_config.get("dataset_name")
    split = dataloader_config.get("split", "test")

    # Use EvaluationExtractor for unified dataset handling
    evaluation_data = EvaluationExtractor.extract(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        split=split,
        limit=limit,
    )

    # Transform evaluation format to include relevant_doc_ids
    queries: list[dict[str, Any]] = []
    for item in evaluation_data:
        # Get the first answer if multiple exist; handle empty answers list
        answer = item["answers"][0] if item["answers"] else ""

        queries.append(
            {
                "query": item["query"],
                "answer": answer,
                # Extract doc ID from metadata if available, otherwise empty
                "relevant_doc_ids": (
                    [item["metadata"]["id"]] if "id" in item["metadata"] else []
                ),
            }
        )

    return queries
