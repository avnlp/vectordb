"""Shared dataloader types and exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


DatasetType = Literal["triviaqa", "arc", "popqa", "factscore", "earnings_calls"]


class DataloaderError(Exception):
    """Base dataloader exception."""


class UnsupportedDatasetError(DataloaderError, ValueError):
    """Raised when an unknown dataset type is requested."""


class DatasetLoadError(DataloaderError):
    """Raised when loading a source dataset fails."""


class DatasetValidationError(DataloaderError, ValueError):
    """Raised when source data does not match expected schema."""


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Normalized dataset document record.

    Args:
        text: Document content to index.
        metadata: Dataset-specific metadata used for evaluation/filtering.
    """

    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    """Evaluation query extracted from normalized records.

    Args:
        query: User/evaluation question.
        answers: Ground-truth answers.
        relevant_doc_ids: Optional IDs of known relevant docs.
        metadata: Additional metadata excluding query/answers.
    """

    query: str
    answers: list[str]
    relevant_doc_ids: list[str]
    metadata: dict[str, Any]
