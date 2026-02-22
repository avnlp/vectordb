# Hybrid Indexing (LangChain)

Hybrid indexing combines dense vector embeddings with sparse (keyword-based) representations to deliver search results that capture both semantic meaning and exact term matches. By blending these two retrieval signals, hybrid search can outperform either approach alone, particularly on queries that mix conceptual intent with specific terminology.

Each database has a dedicated indexing and search pipeline. At query time, the two result sets are fused using either the database's native hybrid ranking mechanism or a client-side fusion strategy. Fusion strategies vary by database: Qdrant, Milvus, and Pinecone use Reciprocal Rank Fusion (RRF), while Weaviate uses a configurable alpha parameter for weighted blending.

## Overview

- Dual embedding during indexing: dense vectors from sentence-transformers and sparse vectors from SPLADE-based sparse embedder
- Native hybrid search on databases that support it, with client-side fusion as a fallback
- Fusion strategies: RRF (Qdrant, Milvus, Pinecone) or alpha-weighted blend (Weaviate)
- Optional RAG answer generation using an LLM (Groq or OpenAI-compatible endpoints)
- Configuration-driven through YAML files with environment variable substitution
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing

The indexing pipeline loads a dataset, generates dense embeddings using a sentence-transformer model, and generates sparse embeddings using a SPLADE-based sparse embedder. Both embedding types are stored together in the vector database. For databases with native hybrid support, vectors are stored in backend-specific formats (for example, Pinecone `sparse_values`). In the Milvus pipeline, records are inserted as Haystack `Document` objects and Milvus auto-generates `INT64` primary keys (`auto_id=True`). The Weaviate pipeline indexes both dense vectors and document text, because Weaviate computes BM25 scores internally at query time rather than accepting external sparse embeddings.

### Search

The search pipeline embeds the query using both the dense and sparse models. It then issues a hybrid search request to the database, which combines dense nearest-neighbor results with sparse keyword-matching results. Qdrant, Pinecone, and Milvus support native hybrid queries that fuse the results server-side using Reciprocal Rank Fusion (RRF). Weaviate blends its internal BM25 scores with dense vector similarity using a configurable alpha parameter. Chroma does not natively support hybrid search, so the pipeline emulates hybrid behavior through client-side fusion of independent dense and sparse result sets.

## Supported Databases

| Database | Hybrid Type | Notes |
|----------|-------------|-------|
| Pinecone | Native hybrid search | Sparse stored in sparse_values format; uses RRF fusion |
| Weaviate | Native BM25 + vector blend | Alpha parameter controls blend weight; BM25 computed internally |
| Milvus | Native hybrid search | Stores sparse vectors in dedicated field; uses RRF fusion |
| Qdrant | Native hybrid search | Uses named sparse vector fields; uses RRF fusion |
| Chroma | Not natively supported | Emulates hybrid via client-side fusion of dense and sparse results |

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
  index_name: "lc-hybrid-triviaqa"
  namespace: ""
  dimension: 384
  metric: "cosine"
  recreate: false
  alpha: 0.5

search:
  top_k: 10

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  name: "lc_hybrid_indexing_pinecone"
  level: "INFO"
```

## Directory Structure

```
hybrid_indexing/
├── __init__.py
├── README.md
├── indexing/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone hybrid indexing
│   ├── weaviate.py                    # Weaviate hybrid indexing (dense + BM25 internal)
│   ├── chroma.py                      # Chroma hybrid indexing
│   ├── milvus.py                      # Milvus hybrid indexing
│   └── qdrant.py                      # Qdrant hybrid indexing
├── search/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone hybrid search
│   ├── weaviate.py                    # Weaviate hybrid search (BM25 + vector)
│   ├── chroma.py                      # Chroma hybrid search with client-side fusion
│   ├── milvus.py                      # Milvus hybrid search
│   └── qdrant.py                      # Qdrant hybrid search
└── configs/                           # 25 YAML configs (5 databases x 5 datasets)
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
- `src/vectordb/langchain/sparse_indexing/` - Sparse-only keyword search pipelines
- `src/vectordb/langchain/utils/` - Shared utilities for configuration, embedding, data loading, and fusion
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
