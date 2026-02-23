# Metadata Filtering

Metadata filtering adds structured, field-level constraints to vector search queries. Rather than relying solely on vector similarity, this module allows narrowing results by document attributes such as category, date, score, or any other metadata field stored alongside the vectors. Filters are defined declaratively in YAML configuration and translated into each database's native filter expression format at runtime.

The module includes built-in timing instrumentation that measures the cost of pre-filtering, vector search, and overall query execution. Selectivity metrics track what fraction of the total document set passes the filter conditions, providing insight into filter effectiveness and query performance.

## Overview

- Structured filtering with operators including equality, comparison, membership, range, and substring matching
- Server-side filter execution where the database supports it, with client-side fallback
- Per-database filter expression builders that translate a common filter specification into native query syntax
- Timing metrics for pre-filter, vector search, and total execution time
- Selectivity analysis to estimate the fraction of documents matching a filter
- Declarative filter schema and test conditions defined in YAML configuration
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing Phase

The indexing pipeline loads a dataset, generates dense embeddings for each document, and stores the vectors along with their metadata fields in the target database. The metadata schema is defined in the configuration, specifying which fields exist, their data types, and which filter operators are valid for each field. Each database-specific indexing module handles collection creation with the appropriate schema, embedding generation, and batch insertion.

### Search Phase

The search pipeline translates the filter conditions from the configuration into the database's native filter expression format. Each database uses a specialized expression builder: for example, the Milvus builder produces string expressions like `category == "science" and score >= 0.9`, while the Pinecone builder produces MongoDB-style dictionaries. The pipeline then executes a filtered vector search, retrieving only documents that satisfy both the metadata constraints and the vector similarity criteria. Timing metrics are captured at each stage to measure filter overhead and search latency.

### Supported Filter Operators

| Operator | String Fields | Numeric Fields | Description |
|----------|---------------|----------------|-------------|
| eq | Yes | Yes | Equals |
| in | Yes | Yes | Value in a given list |
| contains | Yes | No | Substring match |
| gt | No | Yes | Greater than |
| gte | No | Yes | Greater than or equal |
| lt | No | Yes | Less than |
| lte | No | Yes | Less than or equal |
| range | No | Yes | Between a minimum and maximum value |

## Supported Databases

| Database | Status | Native Filter Format |
|----------|--------|---------------------|
| Pinecone | Supported | MongoDB-style dictionary |
| Weaviate | Supported | GraphQL where clause |
| Chroma | Supported | MongoDB-style dictionary |
| Milvus | Supported | String expression |
| Qdrant | Supported | Qdrant filter model objects |

## Configuration

Each pipeline is driven by a YAML configuration file. Below is an example showing the key sections:

```yaml
milvus:
  host: "${MILVUS_HOST:-localhost}"
  port: "${MILVUS_PORT:-19530}"

collection:
  name: "arc_metadata_filtered"
  description: "ARC dataset with metadata filtering"

dataloader:
  type: "haystack"
  dataset_name: "ai2_arc"
  split: "test"
  limit: 1000

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32

metadata_filtering:
  test_query: "What is the main function of mitochondria?"

  schema:
    - field: "category"
      type: "string"
      operators: ["eq", "in", "contains"]
      description: "Document category"
    - field: "score"
      type: "float"
      operators: ["eq", "gt", "gte", "lt", "lte", "range"]
      description: "Relevance score"

  test_filters:
    - name: "high_confidence_science"
      description: "Science documents with high confidence"
      conditions:
        - field: "category"
          operator: "eq"
          value: "science"
        - field: "score"
          operator: "gte"
          value: 0.9

logging:
  level: "INFO"
  name: "metadata_filtering_pipeline"
```

## Directory Structure

```
metadata_filtering/
├── __init__.py                        # Package exports for all pipelines and types
├── README.md
├── base.py                            # Abstract base class for filtering pipelines
├── vectordb_pipeline_type.py          # Core types: filter fields, conditions, specs, builders
├── common/                            # Shared utilities
│   ├── __init__.py
│   ├── config.py                      # Configuration loading and validation
│   ├── dataloader.py                  # Document loading from dataset factory
│   ├── embeddings.py                  # Dense embedder factory
│   ├── filters.py                     # Filter parsing and validation
│   ├── rag.py                         # Optional RAG answer generation
│   ├── timer.py                       # Timing instrumentation
│   └── types.py                       # Data classes for filters, timing, and results
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py
│   ├── weaviate.py
│   ├── chroma.py
│   ├── milvus.py
│   └── qdrant.py
├── search/                            # Database-specific search pipelines
│   ├── __init__.py
│   ├── pinecone.py
│   ├── weaviate.py
│   ├── chroma.py
│   ├── milvus.py
│   └── qdrant.py
├── pinecone.py                        # Legacy Pinecone filtering pipeline
├── weaviate.py                        # Legacy Weaviate filtering pipeline
├── chroma.py                          # Legacy Chroma filtering pipeline
├── milvus.py                          # Legacy Milvus filtering pipeline
├── qdrant.py                          # Legacy Qdrant filtering pipeline
└── configs/                           # 25 YAML configs (5 databases x 5 datasets)
    ├── milvus_triviaqa.yaml
    ├── milvus_arc.yaml
    ├── pinecone_triviaqa.yaml
    ├── qdrant_triviaqa.yaml
    ├── weaviate_triviaqa.yaml
    ├── chroma_triviaqa.yaml
    └── ...                            # Remaining database-dataset combinations
```

## Related Modules

- `src/vectordb/haystack/semantic_search/` - Dense-only semantic search pipelines (can be combined with filtering)
- `src/vectordb/haystack/hybrid_indexing/` - Hybrid dense-plus-sparse search pipelines
- `src/vectordb/haystack/utils/` - Shared utilities for configuration, embedding, and data loading
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
