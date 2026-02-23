# Namespaces

This module provides logical data separation and organization through namespace, collection, and partition management across all five supported vector databases. It enables CRUD operations on namespaces, supports indexing documents into separate namespaces, querying within or across namespaces, and collecting statistics on namespace contents.

Each database uses its native mechanism for data isolation: Pinecone provides native namespace parameters, Weaviate uses multi-tenancy, Milvus uses partition key fields, Qdrant uses payload-based filtering, and Chroma uses separate collections per namespace. The module abstracts these differences behind a unified pipeline interface with consistent configuration, typed results, and shared error handling.

## Overview

- Creates, lists, queries, and deletes namespaces using each database's native isolation mechanism
- Indexes documents from configurable datasets into separate namespaces (such as training data in one namespace and test data in another)
- Supports cross-namespace search to compare query results across multiple namespaces side by side
- Provides namespace statistics including document counts and timing metrics
- Defines naming conventions and generators for consistent namespace identification
- Includes typed result objects, custom exceptions, and timing metric collection
- All behavior is driven by YAML configuration files with environment variable substitution

## How It Works

### Namespace Creation and Management

Each database pipeline provides methods to create, list, check existence of, and delete namespaces. The underlying mechanism differs by database -- Pinecone creates namespaces implicitly on first upsert, Weaviate creates tenants within a collection, Milvus creates partitions, Qdrant tags documents with a namespace payload field, and Chroma creates a separate collection for each namespace.

### Multi-Namespace Indexing

The module supports indexing different dataset splits into different namespaces within a single pipeline run. For example, a configuration can specify that the training split goes into one namespace while the validation split goes into another. Documents are loaded, embedded, and written to the appropriate namespace based on the configuration.

### Cross-Namespace Search

When cross-namespace search is enabled, the pipeline runs the same query against multiple namespaces and returns comparative results. This is useful for comparing retrieval quality across dataset splits, evaluating different embedding strategies stored in separate namespaces, or testing against development and production data side by side.

### Statistics and Metrics

The module collects timing metrics for indexing and search operations, tracks document counts per namespace, and supports structured logging of results. Namespace statistics are returned as typed dataclass objects for programmatic use.

## Supported Databases

| Database | Isolation Mechanism | Notes |
|----------|-------------------|-------|
| Pinecone | Native namespaces | Zero-cost namespace creation within a shared index |
| Weaviate | Multi-tenancy (tenants within collections) | Schema-based with tenant activation and deactivation |
| Milvus | Partition key fields | Partitions within a single collection |
| Qdrant | Payload-based filtering | Namespace identifier stored as a payload field |
| Chroma | Separate collections | One collection per namespace |

## Configuration

Configuration is stored in YAML files under the `configs/` directory, with one file per database-dataset combination. Below is an example:

```yaml
pipeline:
  name: "pinecone_triviaqa_namespaces"
  description: "Pinecone namespace pipeline for TriviaQA dataset"
  database: "pinecone"
  dataset: "triviaqa"

pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "triviaqa-namespaces"
  cloud: "aws"
  region: "us-east-1"
  metric: "cosine"

collection:
  name: "triviaqa_namespaces"

namespaces:
  strategy: "split"
  prefix: ""
  definitions:
    - name: "triviaqa_train"
      split: "train"
      description: "TriviaQA training set"
    - name: "triviaqa_validation"
      split: "validation"
      description: "TriviaQA validation set"

dataloader:
  type: "haystack"
  class: "TriviaQADataloader"
  dataset_name: "trivia_qa"
  config: "rc"
  limit: 1000

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  dimension: 1024
  batch_size: 32

cross_namespace:
  enabled: true
  compare_namespaces: ["triviaqa_train", "triviaqa_validation"]
  top_k: 10

metrics:
  collect_timing: true
  collect_stats: true
  log_results: true

logging:
  level: "INFO"
  name: "pinecone_triviaqa_namespaces"
```

## Directory Structure

```
namespaces/
├── __init__.py                  # Package exports for pipelines, types, and utilities
├── README.md                    # This file
├── types.py                     # Enums, dataclasses, exceptions, and utility classes
├── chroma_collections.py        # Chroma collection-based namespace helpers
├── chroma_namespaces.py         # Chroma namespace pipeline
├── milvus_namespaces.py         # Milvus partition-based namespace pipeline
├── pinecone_namespaces.py       # Pinecone native namespace pipeline
├── qdrant_namespaces.py         # Qdrant payload-filtered namespace pipeline
├── weaviate_namespaces.py       # Weaviate tenant-based namespace pipeline
├── utils/                       # Shared utilities
│   ├── __init__.py
│   ├── config.py                # Configuration loading
│   ├── data.py                  # Dataset loading utilities
│   ├── embeddings.py            # Embedding model wrappers
│   └── timing.py                # Operation timing collection
└── configs/                     # 25 YAML configs (5 databases x 5 datasets)
    ├── chroma_arc.yaml
    ├── chroma_earnings_calls.yaml
    ├── ...
    ├── weaviate_popqa.yaml
    └── weaviate_triviaqa.yaml
```

## Related Modules

- `src/vectordb/haystack/multi_tenancy/` - Full multi-tenancy with tenant lifecycle management and RAG
- `src/vectordb/haystack/dense_indexing/` - Dense embedding indexing without namespace separation
- `src/vectordb/dataloaders/haystack/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and Earnings Calls
