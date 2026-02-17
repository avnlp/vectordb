# Dataloaders

Dataset loaders for loading, normalizing, and converting HuggingFace datasets into framework-specific document formats for indexing and evaluation across the VectorDB toolkit.

## Overview

This module provides a unified interface for loading five evaluation datasets from HuggingFace Hub. Each dataset loader normalizes raw rows into a common `DatasetRecord` format, which can then be converted to Haystack or LangChain documents for consumption by downstream pipelines. The module also extracts deduplicated evaluation queries from loaded records, enabling retrieval benchmarking without coupling to any specific framework or database.

- Abstract base class defining the dataset loading contract with validation, streaming, and record limiting
- Five dataset-specific loaders, each handling schema differences and row expansion
- Catalog-based factory for creating loaders by name without importing concrete classes
- Bidirectional document converters producing Haystack and LangChain document objects
- Evaluation query extraction with normalization and deduplication

## How It Works

All dataset loaders extend `BaseDatasetLoader`, which enforces a two-step contract: first load the raw dataset iterable from HuggingFace, then parse each row into one or more normalized `DatasetRecord` instances. The base class handles record limiting, error wrapping, and validation, while subclasses implement the dataset-specific parsing logic.

The `DataloaderCatalog` provides a factory interface for creating loaders by dataset name, removing the need to import individual loader classes. Once loaded, a `LoadedDataset` wrapper exposes conversion methods (`to_haystack()`, `to_langchain()`) and evaluation query extraction (`evaluation_queries()`), allowing pipelines to consume data in whatever format they require.

The `EvaluationExtractor` scans loaded records for question or entity fields, normalizes and deduplicates them, and pairs each with its ground-truth answers and relevant document IDs. This produces a clean set of `EvaluationQuery` objects for retrieval benchmarking.

## Supported Datasets

| Dataset | Loader Class | HuggingFace ID | Default Split | Description |
|---------|-------------|----------------|---------------|-------------|
| TriviaQA | `TriviaQALoader` | `trivia_qa` | `test` | Question-answering with evidence documents; rows expand into multiple records |
| ARC | `ARCLoader` | `ai2_arc` | `validation` | AI2 Reasoning Challenge with multiple-choice science questions |
| PopQA | `PopQALoader` | `akariasai/PopQA` | `test` | Entity-centric factual questions from Wikipedia |
| FactScore | `FactScoreLoader` | `dskar/FActScore` | `test` | Fact verification with entity-level knowledge passages |
| Earnings Calls | `EarningsCallsLoader` | `lamini/earnings-calls-qa` | `train` | Financial earnings call transcripts with question-answer pairs |

## Directory Structure

```
dataloaders/
    __init__.py              # Package exports for all loaders, types, and exceptions
    base.py                  # Abstract base class defining the dataset loading contract
    catalog.py               # Factory for creating loaders by dataset name
    converters.py            # Bidirectional converters to Haystack and LangChain documents
    dataset.py               # LoadedDataset wrapper with conversion and evaluation methods
    evaluation.py            # Deduplicated evaluation query extraction from records
    types.py                 # Shared types (DatasetRecord, EvaluationQuery) and exceptions
    datasets/
        __init__.py
        triviaqa.py          # TriviaQA loader with row-to-multi-record expansion
        arc.py               # ARC Challenge loader with multiple-choice parsing
        popqa.py             # PopQA loader with entity-centric normalization
        factscore.py         # FactScore loader with knowledge passage extraction
        earnings_calls.py    # Earnings call loader with transcript Q&A parsing
```

## Related Modules

- [`utils/`](../utils/) - Shared core utilities for configuration loading, document conversion, and logging
- [`haystack/`](../haystack/) - Haystack integration pipelines that consume `to_haystack()` output
- [`langchain/`](../langchain/) - LangChain integration pipelines that consume `to_langchain()` output
