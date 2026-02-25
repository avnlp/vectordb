# Maximal Marginal Relevance (LangChain)

Maximal Marginal Relevance (MMR) is a diversity-aware retrieval strategy that balances relevance to the query with diversity among the returned documents. Standard vector search often returns near-duplicate results when the top candidates cluster around the same content. MMR addresses this by penalizing candidates that are too similar to documents already selected, ensuring that the final result set covers a broader range of information.

The pipeline implements a two-phase approach: first, it over-fetches a larger candidate set from the vector database using standard dense similarity search, then it applies the MMR reranking algorithm to select a smaller, more diverse subset. The lambda parameter controls the trade-off, where higher values favor relevance and lower values favor diversity.

## Overview

- Diversity-aware retrieval using the MMR reranking algorithm
- Over-fetches candidates from the vector database to provide a broad pool for reranking
- Lambda parameter controls the relevance-versus-diversity trade-off (0 = maximum diversity, 1 = maximum relevance)
- Greedy selection: the highest-relevance document is chosen first, then each subsequent document is scored with a penalty for similarity to already-selected documents
- Optional RAG answer generation using an LLM (Groq or OpenAI-compatible endpoints)
- Configuration-driven through YAML files with environment variable substitution
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing

The indexing pipeline loads a dataset, generates dense embeddings using a sentence-transformer model, creates or recreates the target collection in the vector database, and upserts all embedded documents. The indexing phase is identical to standard dense indexing, as MMR reranking is applied only at search time. Each database has a dedicated indexing implementation that handles connection setup and collection management according to its specific API.

### Search

The search pipeline embeds the incoming query using the same dense model used during indexing. It retrieves a larger-than-needed candidate set from the vector database (controlled by the top_k parameter). The candidate embeddings and the query embedding are then passed to the MMR algorithm, which greedily selects documents by scoring each candidate as:

```
MMR = lambda * similarity(query, doc) - (1 - lambda) * max_similarity(doc, selected_docs)
```

The algorithm selects mmr_k documents from the candidate pool, returning a result set that is both relevant to the query and internally diverse. When RAG is enabled, the MMR-selected documents are passed as context to an LLM for answer generation.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | Uses namespaces for logical partitioning |
| Weaviate | Supported | Uses collections for organization |
| Chroma | Supported | Lightweight local or client-server deployment |
| Milvus | Supported | Uses collections with configurable metrics |
| Qdrant | Supported | Supports both local and server deployments |

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
  index_name: "lc-mmr-triviaqa"
  namespace: ""
  dimension: 384
  metric: "cosine"
  recreate: false

search:
  top_k: 20                  # Number of candidates to over-fetch
  mmr_k: 5                   # Number of results after MMR reranking
  lambda_param: 0.5          # Relevance vs diversity trade-off

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  name: "lc_mmr_pinecone"
  level: "INFO"
```

## Directory Structure

```
mmr/
├── __init__.py
├── README.md
├── indexing/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone MMR indexing
│   ├── weaviate.py                    # Weaviate MMR indexing
│   ├── chroma.py                      # Chroma MMR indexing
│   ├── milvus.py                      # Milvus MMR indexing
│   └── qdrant.py                      # Qdrant MMR indexing
├── search/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone MMR search
│   ├── weaviate.py                    # Weaviate MMR search
│   ├── chroma.py                      # Chroma MMR search
│   ├── milvus.py                      # Milvus MMR search
│   └── qdrant.py                      # Qdrant MMR search
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

- `src/vectordb/langchain/semantic_search/` - Dense-only semantic search pipelines (without diversity reranking)
- `src/vectordb/langchain/reranking/` - Two-stage retrieval with cross-encoder reranking
- `src/vectordb/langchain/utils/` - Shared utilities for configuration loading, embedding, data loading, MMR computation, and RAG generation
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
