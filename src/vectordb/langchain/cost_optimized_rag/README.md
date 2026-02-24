# Cost-Optimized RAG

Production retrieval-augmented generation pipelines designed with resource awareness for managed vector database services that charge by request or compute unit. The pipelines apply cost optimization strategies including result caching to avoid repeated searches, batch processing for efficient embedding generation, pre-filtering to narrow the search space before vector operations, and cost monitoring to track API calls and token consumption.

This module provides a balance between retrieval quality and operational cost, making it suitable for high-volume production deployments where efficiency matters. All behavior is controlled through YAML configuration files with environment variable substitution for secrets and deployment-specific parameters.

## Overview

- Hybrid search using Weaviate's native BM25 + dense vector fusion
- Single dense embedding API call per query for cost efficiency
- Configurable alpha parameter to balance vector vs. keyword search
- Optional RAG generation with LLM for answer synthesis
- Lazy initialization of expensive components like language models
- Support for all five vector databases with database-specific optimizations
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing Phase

The indexing pipeline loads documents from a configured dataset, embeds them using a sentence transformer model, and writes them to the target vector database collection. Collections are created with configurable vector parameters including dimensionality, distance metric, and optional quantization settings. Payload indexes can be defined in configuration to enable efficient metadata filtering at search time, reducing the number of vectors scanned per query. Documents are upserted in configurable batch sizes to manage memory and network overhead.

### Search Phase

The search pipeline embeds the incoming query using a dense embedding model and executes a hybrid search against the Weaviate database. The hybrid search combines dense vector similarity with BM25 keyword matching in a single query, controlled by the alpha parameter. Retrieved documents are passed to a language model for answer generation when RAG is enabled. Cost metrics are collected throughout the pipeline execution for monitoring and optimization.

### Cost Optimization Strategies

**Hybrid Search** leverages Weaviate's native hybrid search capability that combines dense semantic vectors with BM25 lexical search in a single query. This eliminates the need for separate sparse embedding generation and client-side fusion, reducing computational overhead.

**Single API Call** generates only one dense embedding per query using a cost-effective API-based model. The sparse component is handled by Weaviate's built-in BM25, avoiding additional embedding API calls.

**Configurable Alpha** allows tuning the balance between vector search (semantic matching) and BM25 (keyword matching) to optimize for both quality and cost based on use case requirements.

**Lazy Initialization** defers the creation of expensive components like language models until they are actually needed. This reduces startup time and resource consumption for pipelines that may not always require answer generation.

## Supported Databases

| Database | Status | Cost Optimization Notes |
|----------|--------|-------------------------|
| Pinecone | Supported | Hybrid search with RRF fusion, batch upserts |
| Weaviate | Supported | Native hybrid search (BM25 + vector), alpha parameter tuning |
| Chroma | Supported | Client-side hybrid search with RRF fusion |
| Milvus | Supported | Native hybrid search with RRF fusion |
| Qdrant | Supported | Native hybrid search with RRF fusion |

## Configuration

Each database has per-dataset YAML configuration files organized by database and dataset. The configuration controls all standard pipeline parameters including embeddings, chunking, search, and RAG generation.

```yaml
dataloader:
  type: "triviaqa"
  split: "test"
  limit: 100
  use_text_splitter: false

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32  # Batch size for embedding generation

chunking:
  chunk_size: 1000
  chunk_overlap: 200
  separators:
    - "\n\n"
    - "\n"
    - " "
    - ""

pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "lc-cost-optimized-rag-triviaqa"
  namespace: ""
  dimension: 384
  metric: "cosine"
  recreate: false

search:
  top_k: 10
  alpha: 0.5  # Weaviate hybrid search parameter (1.0 = vector only, 0.0 = BM25 only)

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  name: "lc_cost_optimized_rag_pinecone"
  level: "INFO"
```

### Configuration Sections

- **dataloader**: Dataset source configuration including type, split, and document limit
- **embeddings**: Embedding model settings with batch size for efficient processing
- **chunking**: Document splitting parameters (chunk size, overlap, separators)
- **<database>**: Database-specific connection settings (pinecone, weaviate, chroma, milvus, qdrant)
- **search**: Search parameters including top_k results and alpha for Weaviate hybrid search
- **rag**: Optional RAG generation with LLM model, API key, and token limits
- **logging**: Logging configuration with custom name and log level

### Cost Optimization Notes

The cost optimization is implemented in the pipeline code itself rather than through configuration flags:

- **Hybrid Search**: Combines dense (API-based) embeddings with BM25 lexical search using Weaviate's native hybrid search
- **Weaviate Native Hybrid**: Uses Weaviate's built-in BM25 + vector fusion with configurable alpha parameter
- **Optional RAG**: The `rag.enabled` flag controls whether to invoke the LLM for answer generation
- **Batch Processing**: Embedding batch sizes are configurable to optimize API throughput

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
