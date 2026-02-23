# Hybrid Indexing

Hybrid indexing combines dense vector embeddings with sparse (keyword-based) representations to deliver search results that capture both semantic meaning and exact term matches. By blending these two retrieval signals, hybrid search can outperform either approach alone, particularly on queries that mix conceptual intent with specific terminology.

Each database has a dedicated indexing and search pipeline. At query time, the two result sets are fused using either the database's native hybrid ranking mechanism or a client-side fusion strategy such as Reciprocal Rank Fusion (RRF) or weighted score blending. Fusion strategies vary by database: Qdrant, Milvus, and Pinecone use RRF, while Weaviate uses a configurable alpha parameter for weighted blending.

## Overview

- Dual embedding during indexing: dense vectors from sentence-transformers and sparse vectors from SPLADE models
- Native hybrid search on databases that support it, with client-side RRF or weighted fusion as a fallback
- Fusion strategies: RRF (Qdrant, Milvus, Pinecone) or alpha-weighted blend (Weaviate)
- Evaluation metrics including Recall at k, MRR at k, NDCG at k, and Precision at k
- Configuration-driven through YAML files with environment variable substitution
- No dependency on FastEmbed; all sparse embeddings use native Haystack sentence-transformers components
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing Phase

The indexing pipeline loads a dataset, generates dense embeddings using a sentence-transformer model, and generates sparse embeddings using a SPLADE model. Both embedding types are stored together in the vector database. For databases with native hybrid support, the vectors are stored in their expected format (for example, Pinecone sparse_values or Milvus sparse fields). The Weaviate pipeline indexes only the dense vectors and document text, because Weaviate computes BM25 scores internally at query time rather than accepting external sparse embeddings.

### Search Phase

The search pipeline embeds the query using both the dense and sparse models. It then issues a hybrid search request to the database, which combines dense nearest-neighbor results with sparse keyword-matching results. Qdrant, Pinecone, and Milvus support native hybrid queries that fuse the results server-side using Reciprocal Rank Fusion (RRF). Weaviate blends its internal BM25 scores with dense vector similarity using a configurable alpha parameter. For Chroma, which lacks native hybrid support, the pipeline runs separate dense and sparse searches and merges results using client-side RRF.

## Supported Databases

| Database | Hybrid Type | Sparse Source | Notes |
|----------|-------------|---------------|-------|
| Pinecone | Native hybrid search | SPLADE via sentence-transformers | Sparse stored in sparse_values format; uses RRF fusion |
| Weaviate | Native BM25 + vector blend | Built-in BM25 (no external sparse) | Alpha parameter controls blend weight |
| Chroma | Client-side RRF fusion | SPLADE via sentence-transformers | Runs dense and sparse searches separately |
| Milvus | Native RRF ranking | SPLADE via sentence-transformers | Stores sparse vectors in dedicated field |
| Qdrant | Native hybrid search | SPLADE via sentence-transformers | Uses named sparse vector fields; uses RRF fusion |

Note: Chroma does not natively support hybrid search. The Chroma pipeline emulates hybrid behavior through client-side fusion of independent dense and sparse result sets.

## Configuration

Each pipeline is driven by a YAML configuration file. Below is an example showing the key sections:

```yaml
dataloader:
  type: "triviaqa"         # triviaqa, arc, popqa, factscore, earnings_calls
  split: "test"
  limit: 1000

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  sparse_model: "prithivida/Splade_PP_en_v2"
  device: "cpu"
  batch_size: 32

milvus:                    # Database-specific section (one per config)
  uri: "${MILVUS_URI:-http://localhost:19530}"
  token: "${MILVUS_TOKEN:-}"
  collection_name: "hybrid_triviaqa"
  dimension: 384
  recreate: false
  batch_size: 100

logging:
  name: "hybrid_indexing"
  level: "INFO"
```

## Directory Structure

```
hybrid_indexing/
├── __init__.py                        # Package exports for all pipelines
├── README.md
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone hybrid indexing
│   ├── weaviate.py                    # Weaviate hybrid indexing (dense only, BM25 internal)
│   ├── chroma.py                      # Chroma hybrid indexing
│   ├── milvus.py                      # Milvus hybrid indexing with RRF
│   └── qdrant.py                      # Qdrant hybrid indexing
├── search/                            # Database-specific search pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone hybrid search
│   ├── weaviate.py                    # Weaviate hybrid search (BM25 + vector)
│   ├── chroma.py                      # Chroma hybrid search with RRF fusion
│   ├── milvus.py                      # Milvus hybrid search with RRF
│   └── qdrant.py                      # Qdrant hybrid search
└── configs/                           # 25 YAML configs (5 databases x 5 datasets)
    ├── milvus_triviaqa.yaml
    ├── milvus_arc.yaml
    ├── milvus_popqa.yaml
    ├── milvus_factscore.yaml
    ├── milvus_earnings_calls.yaml
    ├── pinecone_*.yaml                # Same 5 datasets
    ├── qdrant_*.yaml
    ├── weaviate_*.yaml
    └── chroma_*.yaml
```

## Related Modules

- `src/vectordb/haystack/semantic_search/` - Dense-only semantic search pipelines
- `src/vectordb/haystack/sparse_indexing/` - Sparse-only keyword search pipelines
- `src/vectordb/haystack/utils/` - Shared utilities for configuration, embedding, data loading, and fusion
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
