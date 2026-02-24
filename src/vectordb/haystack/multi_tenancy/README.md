# Multi-Tenancy

This module provides tenant-isolated data management for Retrieval-Augmented Generation pipelines. Each supported vector database uses its native isolation mechanism to ensure that documents and queries are scoped to a specific tenant. The module handles tenant lifecycle management including creation, indexing, querying, retrieval, RAG generation, statistics tracking, and deletion.

Tenant isolation strategies vary by database. Pinecone uses namespaces, Weaviate uses native multi-tenancy with per-tenant shards, Milvus uses partition keys with filter-enforced isolation, Qdrant uses tiered multitenancy with payload-based filtering, and Chroma uses tenant and database scoping. Despite these differences, all databases expose a consistent pipeline interface for indexing, retrieval, and RAG operations.

## Overview

- Provides tenant-scoped indexing, retrieval, and RAG pipelines for all five databases
- Uses each database's native isolation mechanism for secure data separation
- Manages the full tenant lifecycle: creation, existence checks, listing, statistics, and deletion
- Resolves tenant context from explicit values, environment variables, or configuration files
- Tracks timing metrics for all operations (embedding, indexing, search, generation)
- Supports environment variable substitution in all configuration files
- Includes a CLI entry point for common tenant management operations

## How It Works

### Tenant Context Resolution

Every operation requires a tenant context, which identifies the target tenant. The context can be created explicitly by specifying a tenant identifier, loaded from the environment (via the TENANT_ID variable), or derived from a configuration file. The tenant context is passed to pipeline constructors and can be overridden on individual operations.

### Isolation Strategies

Each database implements tenant isolation using its strongest available mechanism:

- **Pinecone**: Each tenant maps to a namespace within a shared index. Namespaces provide physical isolation at the serverless level with minimal overhead, supporting hundreds of thousands of tenants per index.
- **Weaviate**: Uses native multi-tenancy with per-tenant shards. Each tenant gets a dedicated shard within the collection, providing physical isolation. Supports automatic tenant creation and tens of thousands of tenants per collection.
- **Milvus**: Uses a partition key field combined with filter expressions. Documents are tagged with a tenant identifier and physically routed to partitions. Filter expressions enforce isolation at query time, supporting millions of tenants.
- **Qdrant**: Implements tiered multitenancy where small tenants share a collection with payload-based filtering and large tenants can be promoted to dedicated collections. Supports mixed tenant sizes efficiently.
- **Chroma**: Uses tenant and database scoping to isolate data. Each tenant maps to a combination of Chroma tenant and database identifiers, providing logical isolation.

### Pipeline Types

The module provides three pipeline types per database. The indexing pipeline ingests documents with tenant-scoped metadata. The retrieval pipeline searches within a tenant's scope and returns ranked documents with scores. The RAG pipeline combines retrieval with language model generation to produce tenant-scoped answers grounded in retrieved context.

## Supported Databases

| Database | Isolation Strategy | Max Tenants | Overhead |
|----------|-------------------|-------------|----------|
| Pinecone | Namespaces | 100k+ per index | Minimal |
| Weaviate | Native multi-tenancy (per-tenant shards) | 10k-100k+ per collection | Moderate |
| Milvus | Partition key with filter expressions | Hundreds of thousands to millions | Minimal |
| Qdrant | Tiered multitenancy (payload filters) | 10k-100k+ with tiered promotion | Minimal for small tenants |
| Chroma | Tenant and database scoping | Thousands to tens of thousands | Minimal |

## Configuration

Each database-dataset combination has a dedicated YAML configuration file. Below is an example showing the key sections:

```yaml
pipeline:
  name: "milvus_arc_multitenancy"
  description: "Milvus multi-tenancy pipeline for ARC dataset"

database:
  type: "milvus"

collection:
  name: "arc_multitenancy"
  description: "ARC dataset with tenant isolation"

multitenancy:
  strategy: "partition_key"
  field_name: "tenant_id"
  partition_key_isolation: true
  num_partitions: 64

embedding:
  model_name: "Qwen/Qwen3-Embedding-0.6B"
  dimension: 1024
  batch_size: 32
  device: "auto"

dataset:
  name: "arc"
  split: "test"
  max_samples: 10000

retrieval:
  top_k: 10
  metric: "cosine"

generator:
  type: "openai"
  model: "gpt-3.5-turbo"
  api_key: "${OPENAI_API_KEY}"
  kwargs:
    temperature: 0.7
    max_tokens: 256

logging:
  level: "INFO"
```

## Directory Structure

```
multi_tenancy/
├── __init__.py                          # Package exports for all pipelines and types
├── README.md                            # This file
├── base.py                              # Abstract base class for multi-tenancy pipelines
├── tenant_context.py                    # Tenant context resolution
├── vectordb_multitenancy_type.py        # Shared type definitions and dataclasses
├── common/                              # Shared utilities
│   ├── __init__.py
│   ├── config.py                        # Configuration loading
│   ├── embeddings.py                    # Embedding model wrappers
│   ├── rag.py                           # RAG generation utilities
│   ├── tenant_context.py                # Tenant context manager
│   ├── timing.py                        # Operation timing metrics
│   └── types.py                         # Result types and enums
├── chroma/                              # Chroma database implementation
│   ├── __init__.py
│   ├── indexing.py                      # Tenant-scoped document indexing
│   └── search.py                        # Tenant-scoped retrieval and RAG
├── milvus/                              # Milvus database implementation
│   ├── __init__.py
│   ├── indexing.py
│   └── search.py
├── pinecone/                            # Pinecone database implementation
│   ├── __init__.py
│   ├── indexing.py
│   └── search.py
├── qdrant/                              # Qdrant database implementation
│   ├── __init__.py
│   ├── indexing.py
│   └── search.py
├── weaviate/                            # Weaviate database implementation
│   ├── __init__.py
│   ├── indexing.py
│   └── search.py
└── configs/                             # 25 YAML configs (5 databases x 5 datasets)
    ├── chroma_arc.yaml
    ├── chroma_earnings_calls.yaml
    ├── ...
    ├── weaviate_popqa.yaml
    └── weaviate_triviaqa.yaml
```

## Related Modules

- `src/vectordb/haystack/namespaces/` - Namespace and partition management for data organization
- `src/vectordb/haystack/rag/` - Standard RAG pipelines without tenant isolation
- `src/vectordb/dataloaders/haystack/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and Earnings Calls
