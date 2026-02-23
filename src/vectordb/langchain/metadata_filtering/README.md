# Metadata Filtering (LangChain)

Metadata filtering extends vector search with structured field-level constraints, enabling queries that combine semantic similarity with precise attribute matching. By applying metadata filters, search results can be narrowed to documents that satisfy specific criteria such as matching a category, falling within a date range, or exceeding a confidence threshold.

The module supports both server-side filtering (where the database applies filters before or during the vector search) and client-side filtering (where results are post-processed after retrieval). Filter conditions are defined in configuration and support operators including equality, comparison, membership, and substring matching.

## Overview

- Field-level metadata filtering combined with dense vector search
- Server-side filter pass-through to databases that support native filtered search
- Client-side post-retrieval filtering for additional conditions defined in configuration
- Configurable filter conditions with operators such as equals, greater-than-or-equal, less-than, in-list, and contains
- Optional RAG answer generation using an LLM (Groq or OpenAI-compatible endpoints)
- Configuration-driven through YAML files with environment variable substitution
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing

The indexing pipeline loads a dataset, generates dense embeddings using a sentence-transformer model, creates or recreates the target collection in the vector database, and upserts all embedded documents along with their metadata fields. Metadata fields stored during indexing become available for filtering at search time. Each database has a dedicated indexing implementation that handles connection setup and collection management according to its specific API.

### Search

The search pipeline embeds the incoming query using the same dense model used during indexing. Filters can be applied at two levels. First, native database filters are passed directly to the vector database query, allowing the database to restrict the candidate set before performing similarity search. Second, if additional filter conditions are defined in the pipeline configuration, client-side filtering removes documents that do not match after retrieval. When RAG is enabled, the filtered documents are passed as context to an LLM for answer generation.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | MongoDB-style filter syntax |
| Weaviate | Supported | GraphQL where-clause filters |
| Chroma | Supported | MongoDB-style filter syntax |
| Milvus | Supported | String expression filters |
| Qdrant | Supported | Structured filter model objects |

## Configuration

Each pipeline is driven by a YAML configuration file using flat naming. Below is an example showing the key sections:

```yaml
dataloader:
  type: "triviaqa"           # triviaqa, arc, popqa, factscore, earnings_calls
  split: "test"
  limit: 100
  use_text_splitter: false

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

pinecone:                     # Database-specific section (one per config)
  api_key: "${PINECONE_API_KEY}"
  index_name: "lc-metadata-filter-triviaqa"
  namespace: ""
  dimension: 384
  metric: "cosine"
  recreate: false

filters:
  conditions:
    - field: "source"
      value: "wikipedia"
      operator: "equals"
    - field: "year"
      value: 2020
      operator: "gte"

search:
  top_k: 10

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  name: "lc_metadata_filtering_pinecone"
  level: "INFO"
```

## Directory Structure

```
metadata_filtering/
├── __init__.py
├── README.md
├── indexing/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone metadata filtering indexing
│   ├── weaviate.py                    # Weaviate metadata filtering indexing
│   ├── chroma.py                      # Chroma metadata filtering indexing
│   ├── milvus.py                      # Milvus metadata filtering indexing
│   └── qdrant.py                      # Qdrant metadata filtering indexing
├── search/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone metadata filtering search
│   ├── weaviate.py                    # Weaviate metadata filtering search
│   ├── chroma.py                      # Chroma metadata filtering search
│   ├── milvus.py                      # Milvus metadata filtering search
│   └── qdrant.py                      # Qdrant metadata filtering search
└── configs/                           # 25 YAML configs (5 databases x 5 datasets)
    ├── __init__.py
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── pinecone_popqa.yaml
    ├── pinecone_factscore.yaml
    ├── pinecone_earnings_calls.yaml
    ├── weaviate_triviaqa.yaml
    ├── ...
    ├── chroma_*.yaml
    ├── milvus_*.yaml
    └── qdrant_*.yaml
```

## Related Modules

- `src/vectordb/langchain/semantic_search/` - Dense-only semantic search pipelines
- `src/vectordb/langchain/hybrid_indexing/` - Combined dense and sparse search pipelines
- `src/vectordb/langchain/utils/` - Shared utilities for configuration loading, embedding, data loading, and document filtering
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
