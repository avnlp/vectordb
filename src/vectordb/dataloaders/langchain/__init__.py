"""LangChain framework-specific dataloaders.

This module provides dataset loaders that integrate with the LangChain
framework (https://python.langchain.com/). These loaders extend the base
dataloader functionality with LangChain-specific features including:

- LangChain Document type integration
- Recursive text splitting using LangChain text splitters
- LLM-based answer summarization using ChatGroq
- Direct compatibility with LangChain vector stores and chains

Architecture:
    Each dataloader in this module wraps the base dataloader functionality
    and adds LangChain-specific processing:

    1. Data Loading: Uses HuggingFace datasets library
    2. Answer Summarization: Uses ChatGroq (for complex datasets)
    3. Document Splitting: Uses RecursiveCharacterTextSplitter for chunking
    4. Output: Returns LangChain Document objects

Datasets Supported:
    - TriviaQA: Open-domain QA with evidence documents
    - ARC: AI2 Reasoning Challenge science QA
    - PopQA: Entity-centric factoid QA
    - FactScore: Fact verification with atomic facts
    - Earnings Calls: Financial transcript Q&A

Dependencies:
    - langchain-core: Core LangChain abstractions
    - langchain-text-splitters: Text splitting utilities
    - langchain-groq: Groq LLM integration
    - datasets: HuggingFace datasets library

Usage:
    >>> from vectordb.dataloaders.langchain import TriviaQADataloader
    >>> from langchain_groq import ChatGroq
    >>>
    >>> generator = ChatGroq(model="llama-3.3-70b-versatile")
    >>> loader = TriviaQADataloader(answer_summary_generator=generator)
    >>> documents = loader.get_documents()

Integration:
    These dataloaders are used by the factory.create_dataloader function
    when the target framework is "langchain". They provide direct
    compatibility with LangChain's VectorStore and Chain APIs.
"""

from vectordb.dataloaders.langchain.arc import ARCDataloader
from vectordb.dataloaders.langchain.earnings_calls import EarningsCallDataloader
from vectordb.dataloaders.langchain.factscore import FactScoreDataloader
from vectordb.dataloaders.langchain.popqa import PopQADataloader
from vectordb.dataloaders.langchain.triviaqa import TriviaQADataloader


__all__ = [
    "ARCDataloader",
    "EarningsCallDataloader",
    "FactScoreDataloader",
    "PopQADataloader",
    "TriviaQADataloader",
]
