# Cost-Optimized RAG

Production retrieval-augmented generation pipelines designed with resource awareness for managed vector database services that charge by request or compute unit. The pipelines apply cost optimization strategies including conditional over-fetching that only retrieves extra candidates when reranking is enabled, lazy initialization of pipeline components to avoid unnecessary resource allocation, and configurable search parameters that control the number of database read units consumed per query.

The module uses Pydantic-validated configuration models to enforce type safety across all pipeline settings, from database connection parameters to quantization and partitioning options. Each database implementation follows a consistent pattern of embedding, optional reranking, and optional LLM-based answer generation, with all behavior controlled through YAML configuration files.

## Overview

- Conditional over-fetching: retrieves 2x candidates only when reranking is active, otherwise fetches exactly the requested count
- Lazy pipeline initialization: RAG generation components are only created when explicitly enabled in configuration
- Configurable search parameters to control database read units and query costs
- Pydantic-validated configuration with environment variable resolution
- Optional cross-encoder reranking controlled by a configuration flag
- Optional RAG answer generation using Groq or OpenAI-compatible language models
- Payload indexing support for efficient metadata filtering
- Partitioning and quantization options for storage cost optimization

## How It Works

### Indexing

The indexing pipeline loads documents from a configured dataset, embeds them using a sentence transformer model, and writes them to the target vector database collection. Collections are created with configurable vector parameters including dimensionality, distance metric, and optional quantization settings. Payload indexes can be defined in configuration to enable efficient metadata filtering at search time, reducing the number of vectors scanned per query. Documents are upserted in configurable batch sizes to manage memory and network overhead.

### Search

The search pipeline embeds the incoming query and executes a vector search against the database. If reranking is enabled in the configuration, the pipeline over-fetches by 2x and applies a cross-encoder similarity ranker to rescore candidates before truncating to the final count. If reranking is disabled, the pipeline fetches exactly the requested number of results, avoiding unnecessary database read units. When RAG generation is enabled and an API key is configured, the retrieved documents are passed through a prompt template and language model to produce a natural language answer alongside the search results.

## Supported Databases

| Database | Indexing Module | Search Module | Notes |
|----------|----------------|---------------|-------|
| Pinecone | `indexing/pinecone_indexer.py` | `search/pinecone_searcher.py` | Serverless managed service |
| Weaviate | `indexing/weaviate_indexer.py` | `search/weaviate_searcher.py` | GraphQL-based queries |
| Chroma | `indexing/chroma_indexer.py` | `search/chroma_searcher.py` | Local or cloud deployment |
| Milvus | `indexing/milvus_indexer.py` | `search/milvus_searcher.py` | Distributed vector database |
| Qdrant | `indexing/qdrant_indexer.py` | `search/qdrant_searcher.py` | Payload filtering support |

## Configuration

Each database has per-dataset YAML configuration files organized in subdirectories. The configuration uses Pydantic models for validation and supports environment variable substitution. Settings cover database connection, embedding model, chunking, indexing (partitions, quantization, vector config, payload indexes), search behavior, reranking, RAG generation, and logging.

```yaml
qdrant:
  host: ${QDRANT_HOST:-localhost}
  port: ${QDRANT_PORT:-6333}
  api_key: ${QDRANT_API_KEY:-}
  https: false

collection:
  name: triviaqa_cost_optimized
  description: TriviaQA with cost-optimized RAG indexing

dataloader:
  type: triviaqa
  dataset_name: trivia_qa
  split: test
  limit: null

embeddings:
  model: Qwen/Qwen3-Embedding-0.6B
  batch_size: 32
  backend: transformers
  cache_embeddings: false

chunking:
  chunk_size: 512
  overlap: 50

indexing:
  partitions:
    enabled: true
    partition_key: dataset_id
  quantization:
    enabled: false
    method: scalar
    compression_ratio: 4.0
  vector_config:
    size: 1024
    distance: Cosine

search:
  top_k: 10
  hybrid_enabled: false
  reranking_enabled: false
  metadata_filtering_enabled: false

reranker:
  use_crossencoder: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 5

generator:
  enabled: true
  provider: groq
  model: llama-3.3-70b-versatile
  api_key: ${GROQ_API_KEY}
  api_base_url: https://api.groq.com/openai/v1
  temperature: 0.7
  max_tokens: 2048

logging:
  level: INFO
  name: qdrant_triviaqa_rag
```

## Directory Structure

```
src/vectordb/haystack/cost_optimized_rag/
├── __init__.py                        # Module docstring and public API
├── base/                              # Shared base classes and utilities
│   ├── __init__.py
│   ├── chunking.py                    # Text chunking configuration
│   ├── config.py                      # Pydantic config models and YAML loader
│   ├── fusion.py                      # Result fusion utilities
│   ├── metrics.py                     # Cost and quality metrics
│   └── sparse_indexing.py             # Sparse indexing support
├── configs/                           # YAML configs organized by database (25 files)
│   ├── qdrant/
│   │   ├── triviaqa.yaml
│   │   ├── arc.yaml
│   │   ├── popqa.yaml
│   │   ├── factscore.yaml
│   │   └── earnings_calls.yaml
│   ├── pinecone/                      # Same structure per database
│   ├── milvus/
│   ├── chroma/
│   └── weaviate/
├── evaluation/                        # Quality and cost evaluation
│   ├── __init__.py
│   └── evaluator.py                   # Evaluation metrics implementation
├── examples/                          # Usage examples
│   └── __init__.py
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── chroma_indexer.py              # Chroma document indexing
│   ├── milvus_indexer.py              # Milvus document indexing
│   ├── pinecone_indexer.py            # Pinecone document indexing
│   ├── qdrant_indexer.py              # Qdrant document indexing
│   └── weaviate_indexer.py            # Weaviate document indexing
├── search/                            # Database-specific search pipelines
│   ├── __init__.py
│   ├── chroma_searcher.py             # Chroma search with cost optimization
│   ├── milvus_searcher.py             # Milvus search with cost optimization
│   ├── pinecone_searcher.py           # Pinecone search with cost optimization
│   ├── qdrant_searcher.py             # Qdrant search with cost optimization
│   └── weaviate_searcher.py           # Weaviate search with cost optimization
├── utils/                             # Shared utility modules
│   ├── __init__.py
│   ├── common.py                      # Logger creation and document loading helpers
│   └── prompt_templates.py            # RAG prompt templates
└── README.md
```

## Related Modules

- `src/vectordb/haystack/rag/` - Standard RAG pipelines without cost optimization focus
- `src/vectordb/haystack/reranking/` - Dedicated reranking pipelines (reranking is optional here)
- `src/vectordb/haystack/dense_indexing/` - Dense indexing shared across features
- `src/vectordb/haystack/contextual_compression/` - Post-retrieval compression as alternative cost strategy
