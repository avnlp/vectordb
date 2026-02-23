# Reranking (LangChain)

Two-stage retrieval pipelines that combine fast dense vector search with cross-encoder reranking for improved relevance. The first stage retrieves a large candidate set using approximate nearest neighbor search over document embeddings. The second stage rescores each candidate using a cross-encoder model that jointly processes the query and document text, producing more accurate relevance judgments than embedding similarity alone.

The initial retrieval over-fetches by a configurable factor (typically three times the final result count) to ensure the reranker has a broad candidate pool. After rescoring, only the top results are returned, balancing precision with latency. All five supported vector databases share a consistent pipeline interface, with database-specific logic isolated in dedicated modules.

## Overview

- Two-stage retrieval: fast dense search followed by cross-encoder reranking
- Over-fetches candidates in the first stage to maximize reranker effectiveness
- Supports multiple cross-encoder models for rescoring
- Separate indexing and search pipelines for independent scaling
- Optional RAG answer generation using an LLM (Groq or OpenAI-compatible endpoints)
- Configuration-driven via YAML files with environment variable support
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing

The indexing pipeline loads documents from a configured dataset, generates dense embeddings using a sentence-transformer model, creates or recreates the target collection in the vector database, and upserts all embedded documents. The indexing phase is identical to standard dense indexing, as reranking is applied only at search time. Each database has a dedicated indexing implementation that handles connection setup and collection management according to its specific API.

### Search

The search pipeline embeds the incoming query using the same sentence-transformer model used during indexing. It then retrieves a larger-than-needed candidate set from the vector database (controlled by the top_k parameter). The over-fetched candidates are passed to a cross-encoder reranker, which jointly encodes each query-document pair to produce a more accurate relevance score than cosine similarity alone. The reranked results are truncated to the final requested count (rerank_k) and returned. When RAG is enabled, the reranked documents are passed as context to an LLM for answer generation.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | Serverless managed service |
| Weaviate | Supported | GraphQL-based queries |
| Chroma | Supported | Local or cloud deployment |
| Milvus | Supported | Distributed vector database |
| Qdrant | Supported | Payload filtering support |

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

reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

pinecone:                     # Database-specific section (one per config)
  api_key: "${PINECONE_API_KEY}"
  index_name: "lc-reranking-triviaqa"
  namespace: ""
  dimension: 384
  metric: "cosine"
  recreate: false

search:
  top_k: 20                  # Number of candidates to over-fetch
  rerank_k: 5                # Number of results after reranking

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  name: "lc_reranking_pinecone"
  level: "INFO"
```

## Directory Structure

```
reranking/
├── __init__.py
├── README.md
├── indexing/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone reranking indexing
│   ├── weaviate.py                    # Weaviate reranking indexing
│   ├── chroma.py                      # Chroma reranking indexing
│   ├── milvus.py                      # Milvus reranking indexing
│   └── qdrant.py                      # Qdrant reranking indexing
├── search/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone search with reranking
│   ├── weaviate.py                    # Weaviate search with reranking
│   ├── chroma.py                      # Chroma search with reranking
│   ├── milvus.py                      # Milvus search with reranking
│   └── qdrant.py                      # Qdrant search with reranking
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

- `src/vectordb/langchain/semantic_search/` - Dense-only semantic search pipelines (without reranking stage)
- `src/vectordb/langchain/mmr/` - Diversity-aware retrieval using Maximal Marginal Relevance
- `src/vectordb/langchain/utils/` - Shared utilities for configuration loading, embedding, data loading, reranker creation, and RAG generation
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
