# Dataloaders

This module provides a unified system for loading, preprocessing, and converting datasets into formats suitable for vector database indexing and retrieval evaluation. The architecture follows a three-tier design: base loaders handle framework-agnostic data acquisition and normalization, framework-specific loaders extend them with Haystack or LangChain document types and text splitting, and shared infrastructure provides registry-based discovery, document conversion, and evaluation support.

All loaders produce a standardized output format consisting of text and metadata dictionaries, regardless of the source dataset. This normalization ensures that any dataset can be used interchangeably across any supported vector database and framework combination.

## Overview

- Five built-in dataset loaders covering open-domain QA, science reasoning, factoid QA, fact verification, and financial transcript analysis
- Three-tier architecture: base loaders, framework-specific loaders, and shared infrastructure
- Registry pattern for unified dataset discovery and instantiation
- Optional LLM-based answer summarization during preprocessing
- Recursive text splitting tuned to each framework's chunking components
- Ground truth extraction for retrieval evaluation
- Standardized output format of text and metadata dictionaries across all datasets

## Supported Datasets

| Dataset | Domain | Description |
|---------|--------|-------------|
| TriviaQA | Open-domain QA | Trivia questions with supporting evidence documents |
| ARC | Science reasoning | AI2 Reasoning Challenge science exam questions |
| PopQA | Entity-centric factoid QA | Factoid questions about popular entities |
| FactScore | Fact verification | Claims paired with supporting or refuting evidence |
| Earnings Calls | Financial transcript QA | Question answering over corporate earnings call transcripts |

## How It Works

The loading pipeline proceeds through several stages. First, a base loader fetches raw data from the source dataset and normalizes it into text and metadata pairs. Optionally, an LLM generator can be injected to produce summarized answers during this stage, which is useful for datasets where raw answers need condensation. Next, the text is split into smaller chunks using recursive splitting strategies appropriate to the target framework. Finally, the chunks are wrapped in framework-specific document objects that are directly compatible with the corresponding indexing pipelines.

The registry module provides a single entry point for discovering and instantiating any supported dataset loader by name. The factory module handles dependency injection of LLM generators and other configurable components. The evaluation module extracts ground truth query-answer pairs from loaded datasets to support retrieval metric computation.

## Directory Structure

```
dataloaders/
    __init__.py            # Package exports
    triviaqa.py            # Base loader for TriviaQA dataset
    arc.py                 # Base loader for ARC dataset
    popqa.py               # Base loader for PopQA dataset
    factscore.py           # Base loader for FactScore dataset
    earnings_calls.py      # Base loader for Earnings Calls dataset
    loaders.py             # Registry pattern for unified dataset discovery and loading
    converters.py          # Document conversion to Haystack or LangChain format
    evaluation.py          # Ground truth query-answer pair extraction
    prompts.py             # LLM prompts for answer summarization
    factory.py             # Factory for creating framework-specific loaders with injected generators
    utils.py               # Helper utilities for data preprocessing
    haystack/              # Haystack-specific dataset loaders
    langchain/             # LangChain-specific dataset loaders
```

## Related Modules

- [haystack/](haystack/) - Haystack-specific loaders that extend the base loaders
- [langchain/](langchain/) - LangChain-specific loaders that extend the base loaders
- [utils/](../utils/) - Shared configuration and evaluation utilities consumed by loaders
