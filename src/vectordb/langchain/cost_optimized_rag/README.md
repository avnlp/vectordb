# Cost-Optimized RAG

Production retrieval-augmented generation pipelines designed with resource awareness for managed vector database services that charge by request or compute unit. The pipelines apply cost optimization strategies including result caching to avoid repeated searches, batch processing for efficient embedding generation, pre-filtering to narrow the search space before vector operations, and cost monitoring to track API calls and token consumption.

This module provides a balance between retrieval quality and operational cost, making it suitable for high-volume production deployments where efficiency matters. All behavior is controlled through YAML configuration files with environment variable substitution for secrets and deployment-specific parameters.

## Overview

- Result caching with configurable TTL to avoid repeated identical searches
- Batch processing for efficient embedding generation and document upserts
- Pre-filtering capabilities to reduce the number of vectors scanned per query
- Cost monitoring and metrics collection for operational visibility
- Configurable trade-offs between quality and cost
- Lazy initialization of expensive components like language models
- Support for all five vector databases with database-specific optimizations
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing Phase

The indexing pipeline loads documents from a configured dataset, embeds them using a sentence transformer model, and writes them to the target vector database collection. Collections are created with configurable vector parameters including dimensionality, distance metric, and optional quantization settings. Payload indexes can be defined in configuration to enable efficient metadata filtering at search time, reducing the number of vectors scanned per query. Documents are upserted in configurable batch sizes to manage memory and network overhead.

### Search Phase

The search pipeline embeds the incoming query and executes a vector search against the database. If caching is enabled and the query has been seen recently, results are returned from cache without hitting the database. For cache misses, the pipeline retrieves documents and optionally applies post-processing such as reranking or compression. Retrieved documents are passed to a language model for answer generation when RAG is enabled. Cost metrics are collected throughout the pipeline execution for monitoring and optimization.

### Cost Optimization Strategies

**Caching** stores recent search results in memory with a configurable time-to-live. Repeated queries within the TTL window are served from cache, eliminating database read costs and embedding computation. The cache uses an LRU eviction policy to manage memory.

**Batch Processing** groups multiple documents or queries for efficient embedding generation. Rather than processing one document at a time, the pipeline processes them in batches to maximize throughput and reduce API call overhead.

**Pre-filtering** applies metadata filters before vector search to reduce the candidate set size. This is particularly effective when queries can be constrained by known attributes like category, date range, or source.

**Lazy Initialization** defers the creation of expensive components like language models until they are actually needed. This reduces startup time and resource consumption for pipelines that may not always require answer generation.

## Supported Databases

| Database | Status | Cost Optimization Notes |
|----------|--------|-------------------------|
| Pinecone | Supported | Namespace-based caching, batch upserts, read unit monitoring |
| Weaviate | Supported | Collection-level caching, GraphQL query optimization |
| Chroma | Supported | Local storage optimization, minimal API costs |
| Milvus | Supported | Partition-based filtering, batch operations |
| Qdrant | Supported | Payload indexing for pre-filtering, batch search |

## Configuration

Each database has per-dataset YAML configuration files organized by database and dataset. The configuration controls caching behavior, batch sizes, pre-filtering settings, cost monitoring, and all standard pipeline parameters.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "cost-optimized-index"
  namespace: ""

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  batch_size: 64  # Larger batches for efficiency

optimization:
  caching:
    enabled: true
    ttl_seconds: 300
    max_size: 1000
  batch_processing:
    enabled: true
    batch_size: 32
  cost_monitoring:
    enabled: true
    track_tokens: true
    track_api_calls: true

pre_filtering:
  enabled: false
  metadata_fields:
    - "category"
    - "date"

search:
  top_k: 10
  reranking_enabled: false

rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 1024  # Limit token usage

logging:
  level: "INFO"
```

## Directory Structure

```
cost_optimized_rag/
├── __init__.py                        # Package exports
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone cost-optimized indexing
│   ├── weaviate.py                    # Weaviate cost-optimized indexing
│   ├── chroma.py                      # Chroma cost-optimized indexing
│   ├── milvus.py                      # Milvus cost-optimized indexing
│   └── qdrant.py                      # Qdrant cost-optimized indexing
├── search/                            # Database-specific search pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone cost-optimized search
│   ├── weaviate.py                    # Weaviate cost-optimized search
│   ├── chroma.py                      # Chroma cost-optimized search
│   ├── milvus.py                      # Milvus cost-optimized search
│   └── qdrant.py                      # Qdrant cost-optimized search
└── configs/                           # YAML configs organized by database
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── weaviate_triviaqa.yaml
    └── ...                            # (25+ config files total)
```

## Related Modules

- `src/vectordb/langchain/semantic_search/` - Standard semantic search without cost optimization
- `src/vectordb/langchain/reranking/` - Dedicated reranking pipelines (optional here)
- `src/vectordb/langchain/contextual_compression/` - Post-retrieval compression as alternative cost strategy
- `src/vectordb/langchain/hybrid_indexing/` - Hybrid search for better retrieval quality
