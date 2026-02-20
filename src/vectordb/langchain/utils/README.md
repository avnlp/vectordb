# LangChain Utilities

This module provides shared utility helpers used across all LangChain feature modules. Each helper encapsulates a specific concern -- configuration loading, embedding creation, document filtering, result merging, reranking, diversification, and RAG generation -- so that feature modules can focus on orchestrating these capabilities rather than reimplementing them.

Compared to the Haystack utilities module, the LangChain utilities include two additional helpers not found in the Haystack counterpart: one for computing maximal marginal relevance scores and one for generating sparse embeddings. These are provided here because LangChain pipelines manage these operations explicitly in user code, whereas the Haystack integration delegates them to built-in pipeline nodes.

## Overview

- Configuration loading with YAML parsing and environment variable resolution
- Dataset integration for loading and chunking documents from supported datasets
- Dense embedding creation using HuggingFace sentence-transformer models
- Sparse embedding creation using SPLADE-based models from sentence-transformers
- Document filtering by metadata fields, nested JSON paths, and custom predicates with support for ten comparison operators
- Result fusion for combining results from multiple queries using reciprocal rank fusion or weighted scoring
- Maximal marginal relevance reranking to balance relevance and diversity in search results
- Semantic diversification using greedy selection or clustering-based approaches
- Cross-encoder reranking for improved relevance scoring of query-document pairs
- RAG answer generation with configurable prompt templates and LLM integration

## How It Works

The configuration loader reads YAML files and resolves environment variable references, providing a dictionary that other helpers consume to initialize their dependencies. The dataset integration helper loads documents from one of five supported datasets (TriviaQA, ARC, PopQA, FactScore, or EarningsCall) and optionally applies recursive text splitting for chunking.

The dense embedding helper wraps HuggingFace sentence-transformer models to produce vector representations for documents and queries. The sparse embedding helper wraps the sentence-transformers sparse encoder (using SPLADE models) to produce sparse token-weight dictionaries, with optional L2 normalization for both document and query embeddings.

The document filter supports ten comparison operators (equals, contains, starts-with, ends-with, greater-than, less-than, greater-or-equal, less-or-equal, in, and not-in) and can traverse nested JSON paths within document metadata. It also supports arbitrary predicate functions for custom filtering logic.

The result merger implements two fusion strategies for combining results from multiple retrieval passes: reciprocal rank fusion and weighted merge scoring. Both strategies support per-result-set weighting and include deduplication by document content or metadata key. The `dedup_key` parameter allows specifying a metadata field (e.g., document ID) for more robust deduplication when documents have similar content but different identifiers.

The maximal marginal relevance helper iteratively selects documents that balance query relevance against redundancy with already-selected documents, controlled by a lambda trade-off parameter. The diversification helper provides both a greedy similarity-threshold approach and a clustering-based approach (using k-means) for selecting diverse document subsets.

The cross-encoder reranker scores query-document pairs jointly using a HuggingFace cross-encoder model and returns documents sorted by relevance, optionally truncated to a top-k limit. The RAG helper formats retrieved documents into a prompt template and generates answers using a language model.

## Directory Structure

```
utils/
    __init__.py              # Package exports for all utility helpers
    config.py                # Configuration loading (re-exports from Haystack utils)
    dataloader.py            # Dataset loading and document chunking
    diversification.py       # Semantic diversification (greedy and clustering-based)
    embeddings.py            # Dense embedding creation with HuggingFace models
    filters.py               # Document filtering by metadata, JSON paths, and predicates
    fusion.py                # Result merging with reciprocal rank fusion and weighted scoring
    mmr.py                   # Maximal marginal relevance reranking (LangChain-specific)
    rag.py                   # RAG prompt formatting and answer generation
    reranker.py              # Cross-encoder reranking with HuggingFace models
    sparse_embeddings.py     # Sparse embedding creation with SPLADE models (LangChain-specific)
```

## Related Modules

- `src/vectordb/haystack/utils/` - Haystack utility helpers (shared configuration loader, no MMR or sparse embedder)
- `src/vectordb/langchain/components/` - Higher-level pipeline components that build on these utilities
- `src/vectordb/utils/` - Core shared utilities (document converters, logging)
- `src/vectordb/dataloaders/langchain/` - LangChain-specific dataset loaders consumed by the dataloader helper
