# Semantic Search (LangChain)

Semantic search provides end-to-end dense vector retrieval pipelines for all five supported vector databases using LangChain. Documents are embedded into dense vectors using a sentence-transformer model, stored in a vector database, and retrieved at query time by computing cosine similarity between a query embedding and the stored document embeddings.

Each database has a dedicated indexing and search pipeline. The search pipeline supports optional retrieval-augmented generation (RAG) to produce natural-language answers grounded in the retrieved documents.

## Overview

- Dense vector indexing using sentence-transformer embedding models
- Similarity search with configurable distance metrics (cosine, euclidean, dot product)
- Optional RAG answer generation using an LLM (Groq or OpenAI-compatible endpoints)
- Fully configuration-driven through YAML files with environment variable substitution
- 25 pre-built configuration files covering all database and dataset combinations
- Optional metadata filter pass-through at query time

## How It Works

### Indexing

The indexing pipeline loads a dataset through the dataloader, converts each record into a LangChain document, and passes the documents through a dense embedding model. The resulting vectors and associated metadata are then upserted into the target vector database. Each database-specific indexing module handles the connection setup, index or collection creation, and batch insertion according to that database's native protocol.

### Search

The search pipeline embeds the incoming query text using the same dense model used during indexing. It then issues a nearest-neighbor search against the vector database, retrieving the top-k most similar documents by cosine similarity. If metadata filters are provided, they are passed to the database to narrow results server-side. When RAG is enabled, the retrieved documents are passed as context to an LLM, which generates a concise answer to the original query.

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
  index_name: "lc-semantic-search-triviaqa"
  namespace: ""
  dimension: 384
  metric: "cosine"
  recreate: false

search:
  top_k: 10

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  name: "lc_semantic_search_pinecone"
  level: "INFO"
```

## Directory Structure

```
semantic_search/
├── __init__.py
├── README.md
├── indexing/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone dense indexing
│   ├── weaviate.py                    # Weaviate dense indexing
│   ├── chroma.py                      # Chroma dense indexing
│   ├── milvus.py                      # Milvus dense indexing
│   └── qdrant.py                      # Qdrant dense indexing
├── search/
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone semantic search
│   ├── weaviate.py                    # Weaviate semantic search
│   ├── chroma.py                      # Chroma semantic search
│   ├── milvus.py                      # Milvus semantic search
│   └── qdrant.py                      # Qdrant semantic search
└── configs/                           # 25 YAML configs (5 databases x 5 datasets)
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── pinecone_popqa.yaml
    ├── pinecone_factscore.yaml
    ├── pinecone_earnings_calls.yaml
    ├── weaviate_triviaqa.yaml
    ├── weaviate_arc.yaml
    ├── ...
    ├── chroma_*.yaml
    ├── milvus_*.yaml
    └── qdrant_*.yaml
```

## Related Modules

- `src/vectordb/langchain/utils/` - Shared utilities for configuration loading, embedding, data loading, and RAG generation
- `src/vectordb/langchain/hybrid_indexing/` - Combined dense and sparse search pipelines
- `src/vectordb/langchain/sparse_indexing/` - Sparse keyword-based search pipelines
- `src/vectordb/langchain/metadata_filtering/` - Structured field-level filtering pipelines
- `src/vectordb/langchain/mmr/` - Diversity-aware retrieval using Maximal Marginal Relevance
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
