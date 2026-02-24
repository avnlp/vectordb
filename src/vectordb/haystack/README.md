# Haystack Integration

Pipeline components for building retrieval-augmented generation applications using the Haystack framework across five supported vector databases.

## Overview

This module provides over fifteen advanced RAG features, each implemented as a self-contained pipeline that works with any supported database and dataset combination. The design is configuration-driven: all behavior is parameterized through YAML files, eliminating hardcoded connection strings and feature settings. Each feature follows a consistent two-phase pipeline pattern covering both indexing and search.

> **Note:** Vector database wrapper classes (ChromaVectorDB, MilvusVectorDB, etc.) now live in [`databases/`](../databases/) and are shared by both Haystack and LangChain integrations.

## How It Works

Every feature follows a two-phase pattern:

1. **Indexing phase** - Loads documents from a dataset, generates embeddings (dense, sparse, or both), and stores the resulting vectors in the target database.

2. **Search phase** - Embeds the incoming query, retrieves candidates from the database, applies optional post-processing (reranking, compression, diversification, or filtering), and optionally generates an answer using a large language model.

All behavior is parameterized via YAML configuration files located in each feature's `configs/` subdirectory. Each feature maintains 25 configuration files covering every combination of five databases and five datasets (TriviaQA, ARC, PopQA, FactScore, and EarningsCall).

## Supported Databases

| Database | Description |
|----------|-------------|
| Pinecone | Managed vector database with native sparse vector and namespace support |
| Weaviate | Open-source vector search engine with BM25 and vector merge via alpha parameter |
| Chroma | Lightweight embedding database for dense retrieval |
| Milvus | Scalable vector database with hybrid search and partition-based isolation |
| Qdrant | High-performance vector search with collection-based multi-tenancy |

## Module Index

| Module | Description |
|--------|-------------|
| `semantic_search/` | Dense vector retrieval using embedding similarity |
| `hybrid_indexing/` | Combined dense and sparse search with fusion strategies |
| `sparse_indexing/` | Keyword-based sparse vector search |
| `metadata_filtering/` | Structured field-level filtering on document metadata |
| `mmr/` | Diversity-aware ranking using maximal marginal relevance |
| `reranking/` | Two-stage retrieval with cross-encoder reranking |
| `query_enhancement/` | LLM-generated query variations for improved recall |
| `parent_document_retrieval/` | Hierarchical chunking with parent-child document relationships |
| `contextual_compression/` | Post-retrieval document compression to reduce noise |
| `cost_optimized_rag/` | Production-oriented RAG with resource-aware optimizations |
| `diversity_filtering/` | Clustering-based result diversification to reduce redundancy |
| `json_indexing/` | Structured JSON document indexing and retrieval |
| `agentic_rag/` | Self-reflecting retrieval with intelligent query routing |
| `multi_tenancy/` | Tenant-isolated data management across shared infrastructure |
| `namespaces/` | Logical data separation within a single database instance |
| `utils/` | Shared utility helpers for configuration, embeddings, filtering, and more |
| `components/` | Reusable pipeline components for query expansion, fusion, compression, and evaluation |

## Directory Structure

```
haystack/
    semantic_search/
    hybrid_indexing/
    sparse_indexing/
    metadata_filtering/
    mmr/
    reranking/
    query_enhancement/
    parent_document_retrieval/
    contextual_compression/
    cost_optimized_rag/
    diversity_filtering/
    json_indexing/
    agentic_rag/
    multi_tenancy/
    namespaces/
    utils/
    components/
```

## Related Modules

- [`utils/`](../utils/) - Logging, document converters, and shared project-wide utilities.
- [`dataloaders/`](../dataloaders/) - Framework-specific dataset loaders that normalize output for pipeline consumption.
- [`langchain/`](../langchain/) - LangChain integrations providing an alternative framework for the same vector database backends.
