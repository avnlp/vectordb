"""Dataloader module for vector database integrations and RAG evaluation.

This module provides dataset loaders for common RAG evaluation benchmarks including
TriviaQA, ARC, PopQA, FactScore, and Earnings Calls. It supports both Haystack and
LangChain framework integration.

The dataloaders standardize diverse dataset formats into a common structure with
"text" (document content) and "metadata" (annotations, questions, answers) keys.
This enables consistent evaluation across different retrieval pipelines and vector
database backends.

Datasets Supported:
    - TriviaQA: Open-domain question answering with evidence documents from web search.
        Useful for testing retrieval on trivia questions with multiple evidence sources.
    - ARC: AI2 Reasoning Challenge for science question answering.
        Tests reasoning capabilities with multiple-choice science questions.
    - PopQA: Popular factoid QA with entity-centric queries.
        Focuses on knowledge retrieval for well-known entities.
    - FactScore: Fact verification with atomic facts.
        Evaluates factuality at the atomic level using Wikipedia-derived facts.
    - Earnings Calls: Financial transcript Q&A for domain-specific evaluation.
        Tests retrieval on financial documents with temporal and entity metadata.

Architecture:
    The module follows a layered architecture:

    1. Base Dataloaders (arc.py, triviaqa.py, etc.):
       Simple loaders that return standardized dicts. Used for quick data access
       and basic evaluation scenarios.

    2. Framework-Specific Dataloaders (haystack/, langchain/):
       Extended loaders that integrate with framework Document types and
       preprocessing pipelines. Support text splitting and LLM-based answer
       summarization for complex datasets.

    3. Registry Pattern (loaders.py):
       DatasetRegistry provides a unified interface for loading any supported
       dataset by type name, enabling configuration-driven data loading.

    4. Document Conversion (converters.py):
       DocumentConverter transforms standardized dicts to framework-specific
       Document objects (HaystackDocument, LangChainDocument).

    5. Evaluation Support (evaluation.py):
       EvaluationExtractor extracts query-answer pairs in a format suitable
       for RAG evaluation pipelines.

Usage Examples:
    Basic dataset loading:
        >>> from vectordb.dataloaders import DatasetRegistry
        >>> data = DatasetRegistry.load("triviaqa", split="test", limit=100)
        >>> # Returns [{"text": "...", "metadata": {"question": "...",
        ...     "answer": "..."}}]

    Converting to framework documents:
        >>> from vectordb.dataloaders import DocumentConverter
        >>> haystack_docs = DocumentConverter.to_haystack(data)
        >>> langchain_docs = DocumentConverter.to_langchain(data)

    Extracting evaluation queries:
        >>> from vectordb.dataloaders import EvaluationExtractor
        >>> queries = EvaluationExtractor.extract("triviaqa", split="test", limit=50)
        >>> # Returns [{"query": "...", "answers": [...], "metadata": {...}}]

Integration Points:
    - VectorDB indexing pipelines use DocumentConverter to prepare documents
    - RAG evaluation scripts use EvaluationExtractor for ground truth
    - Configuration-driven pipelines use DatasetRegistry for dynamic loading
"""

from vectordb.dataloaders.converters import DocumentConverter
from vectordb.dataloaders.evaluation import EvaluationExtractor
from vectordb.dataloaders.loaders import (
    ARCDataloader,
    DataloaderProtocol,
    DatasetRegistry,
    EarningsCallDataloader,
    FactScoreDataloader,
    PopQADataloader,
    TriviaQADataloader,
)


__all__ = [
    "DatasetRegistry",
    "DocumentConverter",
    "EvaluationExtractor",
    # Protocol for custom dataloader implementations
    "DataloaderProtocol",
    # Individual loaders for domain specific datasets
    "ARCDataloader",
    "EarningsCallDataloader",
    "FactScoreDataloader",
    "PopQADataloader",
    "TriviaQADataloader",
]
