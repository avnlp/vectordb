# Haystack Dataloaders

This module provides Haystack-specific dataset loaders that extend the base dataloaders with Haystack document types, text splitting components, and LLM-based preprocessing. Each loader produces Haystack-native document objects that are directly compatible with Haystack indexing and retrieval pipelines, eliminating the need for manual format conversion before ingestion.

The loaders handle recursive document splitting using Haystack's built-in splitting components and support optional answer summarization through OpenAI-compatible generators. This allows raw dataset content to be transformed into properly chunked, enriched documents ready for vector database indexing in a single step.

## Overview

- Extends base loaders with Haystack document type integration
- Recursive document splitting using Haystack splitting components
- LLM-based answer summarization using OpenAI-compatible generators
- Direct compatibility with Haystack indexing pipelines and document stores
- Five dataset loaders covering all supported datasets

## How It Works

Each Haystack dataloader inherits from its corresponding base loader and adds framework-specific processing. After the base loader fetches and normalizes the raw data, the Haystack loader splits the text into chunks using recursive splitting configured for the target indexing pipeline. When an LLM generator is provided, answer fields are summarized to produce concise, self-contained responses. The final output is a collection of Haystack document objects with text content, metadata, and optional embeddings ready for direct insertion into any supported document store.

## Directory Structure

```
haystack/
    __init__.py            # Package exports
    triviaqa.py            # Haystack loader for TriviaQA dataset
    arc.py                 # Haystack loader for ARC dataset
    popqa.py               # Haystack loader for PopQA dataset
    factscore.py           # Haystack loader for FactScore dataset
    earnings_calls.py      # Haystack loader for Earnings Calls dataset
```

## Related Modules

- [dataloaders/](../) - Base loaders, registry, and shared infrastructure
- [langchain/](../langchain/) - LangChain-specific loaders for the same datasets
- [haystack/](../../haystack/) - Haystack integrations that consume documents produced by these loaders
