# LangChain Dataloaders

This module provides LangChain-specific dataset loaders that extend the base dataloaders with LangChain document types, text splitting components, and LLM-based preprocessing. Each loader produces LangChain-native document objects that are directly compatible with LangChain vector stores, retrieval chains, and RAG pipelines, eliminating the need for manual format conversion before ingestion.

The loaders handle recursive text splitting using LangChain's text splitter utilities and support optional answer summarization through Groq-compatible generators. This allows raw dataset content to be transformed into properly chunked, enriched documents ready for vector database indexing in a single step.

## Overview

- Extends base loaders with LangChain document type integration
- Recursive text splitting using LangChain text splitter utilities
- LLM-based answer summarization using Groq-compatible generators
- Direct compatibility with LangChain vector stores and retrieval chains
- Five dataset loaders covering all supported datasets

## How It Works

Each LangChain dataloader inherits from its corresponding base loader and adds framework-specific processing. After the base loader fetches and normalizes the raw data, the LangChain loader splits the text into chunks using recursive text splitting configured for the target vector store. When an LLM generator is provided, answer fields are summarized to produce concise, self-contained responses. The final output is a collection of LangChain document objects with page content, metadata, and optional embeddings ready for direct insertion into any supported vector store.

## Directory Structure

```
langchain/
    __init__.py            # Package exports
    triviaqa.py            # LangChain loader for TriviaQA dataset
    arc.py                 # LangChain loader for ARC dataset
    popqa.py               # LangChain loader for PopQA dataset
    factscore.py           # LangChain loader for FactScore dataset
    earnings_calls.py      # LangChain loader for Earnings Calls dataset
```

## Related Modules

- [dataloaders/](../) - Base loaders, registry, and shared infrastructure
- [haystack/](../haystack/) - Haystack-specific loaders for the same datasets
- [langchain/](../../langchain/) - LangChain integrations that consume documents produced by these loaders
