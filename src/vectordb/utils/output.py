"""Structured output types for retrieval pipelines.

This module provides standardized dataclasses for representing retrieval results
at various stages of the pipeline lifecycle. These containers ensure consistent
output formats across different vector database implementations and enable
seamless serialization for logging, evaluation, and API responses.

Output Types:
    - RetrievedDocument: Individual document with content, score, and metadata
    - RetrievalOutput: Complete result set from a single query
    - PipelineOutput: Aggregated results from full indexing + retrieval pipeline

Design Notes:
    All output types are implemented as dataclasses for immutability, type safety,
    and automatic __init__, __repr__, and __eq__ generation. Each class provides
    a to_dict() method for JSON serialization and PipelineOutput includes a
    summary() method for human-readable reporting.

Usage:
    >>> from vectordb.utils.output import RetrievedDocument, RetrievalOutput
    >>> doc = RetrievedDocument(
    ...     content="Python is a programming language",
    ...     doc_id="doc-123",
    ...     score=0.95,
    ...     metadata={"source": "wikipedia"},
    ... )
    >>> output = RetrievalOutput(
    ...     query="What is Python?", documents=[doc], top_k=5, latency_ms=45.2
    ... )
    >>> print(output.to_dict())
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedDocument:
    """A single retrieved document with metadata.

    Attributes:
        content: The document text content.
        doc_id: Unique identifier for the document.
        score: Retrieval similarity score.
        metadata: Additional document metadata.
        matched_children: For parent-child retrieval, the matched child chunks.
    """

    content: str
    doc_id: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    matched_children: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all document fields.
        """
        result = {
            "content": self.content,
            "doc_id": self.doc_id,
            "score": self.score,
            "metadata": self.metadata,
        }
        if self.matched_children:
            result["matched_children"] = self.matched_children
        return result


@dataclass
class RetrievalOutput:
    """Complete output from a retrieval operation.

    Attributes:
        query: The input query string.
        documents: List of retrieved documents.
        retrieval_mode: Mode used (with_parents, children_only, context_window).
        top_k: Number of results requested.
        total_retrieved: Actual number of documents retrieved.
        latency_ms: Retrieval latency in milliseconds.
    """

    query: str
    documents: list[RetrievedDocument] = field(default_factory=list)
    retrieval_mode: str = "with_parents"
    top_k: int = 5
    total_retrieved: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with query, documents, and metadata.
        """
        return {
            "query": self.query,
            "documents": [doc.to_dict() for doc in self.documents],
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "total_retrieved": self.total_retrieved,
            "latency_ms": self.latency_ms,
        }


@dataclass
class PipelineOutput:
    """Complete output from running a pipeline with indexing and retrieval.

    Attributes:
        pipeline_name: Name/type of the pipeline.
        database_type: Vector database used.
        dataset_name: Dataset used for indexing.
        index_stats: Statistics from document indexing.
        retrieval_results: List of retrieval outputs for queries.
        evaluation_metrics: Optional evaluation metrics if ground truth available.
    """

    pipeline_name: str
    database_type: str
    dataset_name: str
    index_stats: dict[str, int] = field(default_factory=dict)
    retrieval_results: list[RetrievalOutput] = field(default_factory=list)
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all pipeline output fields.
        """
        return {
            "pipeline_name": self.pipeline_name,
            "database_type": self.database_type,
            "dataset_name": self.dataset_name,
            "index_stats": self.index_stats,
            "retrieval_results": [r.to_dict() for r in self.retrieval_results],
            "evaluation_metrics": self.evaluation_metrics,
        }

    def summary(self) -> str:
        """Generate a human-readable summary of the pipeline output.

        Returns:
            Formatted string summarizing the pipeline run.
        """
        lines = [
            f"Pipeline: {self.pipeline_name}",
            f"Database: {self.database_type}",
            f"Dataset: {self.dataset_name}",
            f"Indexed: {self.index_stats.get('num_documents', 0)} documents",
            f"  Parents: {self.index_stats.get('num_parents', 0)}",
            f"  Children: {self.index_stats.get('num_children', 0)}",
            f"Queries evaluated: {len(self.retrieval_results)}",
        ]

        if self.evaluation_metrics:
            lines.append("Evaluation Metrics:")
            for key, value in self.evaluation_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)
