"""Haystack framework-specific dataloaders.

This module provides dataset loaders that integrate with the Haystack
framework (https://haystack.deepset.ai/). These loaders extend the base
dataloader functionality with Haystack-specific features including:

- Haystack Document type integration
- Recursive document splitting using Haystack components
- LLM-based answer summarization using OpenAIChatGenerator
- Direct compatibility with Haystack indexing pipelines

Architecture:
    Each dataloader in this module wraps the base dataloader functionality
    and adds Haystack-specific processing:

    1. Data Loading: Uses HuggingFace datasets library
    2. Answer Summarization: Uses OpenAIChatGenerator (for complex datasets)
    3. Document Splitting: Uses RecursiveDocumentSplitter for chunking
    4. Output: Returns Haystack Document objects

Datasets Supported:
    - TriviaQA: Open-domain QA with evidence documents
    - ARC: AI2 Reasoning Challenge science QA
    - PopQA: Entity-centric factoid QA
    - FactScore: Fact verification with atomic facts
    - Earnings Calls: Financial transcript Q&A

Dependencies:
    - haystack-ai: Core Haystack framework
    - datasets: HuggingFace datasets library

Usage:
    >>> from vectordb.dataloaders.haystack import TriviaQADataloader
    >>> from haystack.components.generators.chat import OpenAIChatGenerator
    >>>
    >>> generator = OpenAIChatGenerator(model="gpt-4")
    >>> loader = TriviaQADataloader(answer_summary_generator=generator)
    >>> documents = loader.get_documents()

Integration:
    These dataloaders are used by the factory.create_dataloader function
    when the target framework is "haystack". They provide direct
    compatibility with Haystack's DocumentStore and Pipeline APIs.
"""

from vectordb.dataloaders.haystack.arc import ARCDataloader
from vectordb.dataloaders.haystack.earnings_calls import EarningsCallDataloader
from vectordb.dataloaders.haystack.factscore import FactScoreDataloader
from vectordb.dataloaders.haystack.popqa import PopQADataloader
from vectordb.dataloaders.haystack.triviaqa import TriviaQADataloader


__all__ = [
    "ARCDataloader",
    "EarningsCallDataloader",
    "FactScoreDataloader",
    "PopQADataloader",
    "TriviaQADataloader",
]
