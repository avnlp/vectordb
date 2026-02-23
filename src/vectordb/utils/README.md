# Utils

This module contains shared utilities used across both the Haystack and LangChain frameworks within the VectorDB toolkit. These utilities handle cross-cutting concerns such as configuration management, retrieval evaluation, document format conversion, cost tracking, and sparse embedding normalization.

All utilities are designed to be framework-agnostic wherever possible, providing a consistent foundation that both integration layers build upon. Document converters are the primary exception, as each converter translates between a specific database format and the corresponding framework document types.

## Overview

- YAML configuration loading with environment variable resolution
- Six retrieval evaluation metrics with per-query and aggregate computation
- Deterministic document ID generation with a three-level fallback strategy
- Bidirectional document converters for Pinecone, Weaviate, Chroma, and Qdrant
- Sparse embedding normalization across multiple database-specific formats
- Metadata injection and filter construction for tenant isolation

## How It Works

Configuration loading reads YAML files and resolves embedded environment variable references, supporting both required variables and variables with default values. This allows database connection parameters and feature settings to be externalized without hardcoding secrets.

The evaluation module computes recall at k, precision at k, mean reciprocal rank, discounted cumulative gain, normalized discounted cumulative gain, and hit rate. Each metric is calculated per query and then aggregated across the full query set.

Document converters provide bidirectional translation between framework document objects and database-native formats. Each converter handles the specific field mappings, embedding layouts, and metadata structures required by its target database.

The sparse embedding normalizer accepts sparse vectors in any of the supported database formats and converts them to a common representation, enabling cross-database sparse search comparisons.

## Directory Structure

```
utils/
    __init__.py                        # Package exports
    config.py                          # YAML config loading with env var resolution (${VAR} and ${VAR:-default})
    evaluation.py                      # Six retrieval metrics: recall@k, precision@k, MRR, DCG, NDCG, hit rate
    ids.py                             # Three-level ID fallback strategy and bidirectional ID setting
    output.py                          # Structured containers for retrieval results and pipeline output
    logging.py                         # Centralized logger factory with environment-based configuration
    scope.py                           # Metadata injection and filter construction for tenant isolation
    sparse.py                          # Sparse embedding normalization across Pinecone, Milvus, and Qdrant formats
    pinecone_document_converter.py     # Bidirectional conversion between framework and Pinecone formats
    weaviate_document_converter.py     # Bidirectional conversion between framework and Weaviate formats
    chroma_document_converter.py       # Bidirectional conversion between framework and Chroma formats
    qdrant_document_converter.py       # Bidirectional conversion between framework and Qdrant formats
```

## Related Modules

- [dataloaders/](../dataloaders/) - Dataset loaders that consume configuration and produce documents for conversion
- [haystack/](../haystack/) - Haystack integrations that use these utilities for document conversion and evaluation
- [langchain/](../langchain/) - LangChain integrations that use these utilities for document conversion and evaluation
