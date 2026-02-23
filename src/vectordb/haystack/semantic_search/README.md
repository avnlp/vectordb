# Semantic Search

Semantic search provides end-to-end dense vector retrieval pipelines for all five supported vector databases. Documents are embedded into dense vectors using a sentence-transformer model, stored in a vector database, and retrieved at query time by computing cosine similarity between a query embedding and the stored document embeddings.

The module supports optional post-retrieval features including client-side metadata filtering, semantic diversification to reduce redundant results, and retrieval-augmented generation (RAG) to produce natural-language answers grounded in the retrieved documents.

## Overview

- Dense vector indexing using sentence-transformer embedding models
- Similarity search with configurable distance metrics (cosine, euclidean, dot product)
- Client-side metadata filtering with operators such as equality, comparison, and membership
- Semantic diversification to remove near-duplicate results based on a similarity threshold
- Optional RAG answer generation using an LLM (Groq or OpenAI-compatible endpoints)
- Fully configuration-driven through YAML files with environment variable substitution
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing Phase

The indexing pipeline loads a dataset through the dataloader, converts each record into a Haystack document, and passes the documents through a dense embedding model. The resulting vectors and associated metadata are then upserted into the target vector database. Each database-specific indexing module handles the connection setup, index or collection creation, and batch insertion according to that database's native protocol.

### Search Phase

The search pipeline embeds the incoming query text using the same dense model used during indexing. It then issues a nearest-neighbor search against the vector database, retrieving the top-k most similar documents by cosine similarity. If metadata filters are provided, they are applied to narrow results. When semantic diversification is enabled, the pipeline removes documents that are too similar to one another based on a configurable diversity threshold. If RAG is enabled, the retrieved documents are passed as context to an LLM, which generates a concise answer to the original query.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | Uses namespaces for logical partitioning |
| Weaviate | Supported | Uses collections for organization |
| Chroma | Supported | Lightweight local or client-server deployment |
| Milvus | Supported | Uses collections with configurable metrics |
| Qdrant | Supported | Supports both local and server deployments |

## Configuration

Each pipeline is driven by a YAML configuration file. Below is an example showing the key sections:

```yaml
dataloader:
  type: "arc"              # triviaqa, arc, popqa, factscore, earnings_calls
  split: "test"
  limit: 1000

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

pinecone:                  # Database-specific section (one per config)
  api_key: "${PINECONE_API_KEY}"
  index_name: "semantic-arc"
  namespace: ""
  dimension: 384
  metric: "cosine"

search:
  top_k: 10

semantic_diversification:
  enabled: false
  diversity_threshold: 0.7
  max_similar_docs: 2

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  api_base_url: "https://api.groq.com/openai/v1"
  temperature: 0.7
  max_tokens: 2048
```

## Directory Structure

```
semantic_search/
├── __init__.py                        # Package exports for all pipelines
├── README.md
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone dense indexing
│   ├── weaviate.py                    # Weaviate dense indexing
│   ├── chroma.py                      # Chroma dense indexing
│   ├── milvus.py                      # Milvus dense indexing
│   └── qdrant.py                      # Qdrant dense indexing
├── search/                            # Database-specific search pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone semantic search
│   ├── weaviate.py                    # Weaviate semantic search
│   ├── chroma.py                      # Chroma semantic search
│   ├── milvus.py                      # Milvus semantic search
│   └── qdrant.py                      # Qdrant semantic search
└── configs/                           # 25 YAML configs (5 databases x 5 datasets)
    ├── pinecone/
    │   ├── triviaqa.yaml
    │   ├── arc.yaml
    │   ├── popqa.yaml
    │   ├── factscore.yaml
    │   └── earnings_calls.yaml
    ├── weaviate/                       # Same 5 dataset files per database
    ├── chroma/
    ├── milvus/
    └── qdrant/
```

## Related Modules

- `src/vectordb/haystack/utils/` - Shared utilities for configuration loading, embedding, data loading, filtering, diversification, and RAG generation
- `src/vectordb/haystack/hybrid_indexing/` - Hybrid dense-plus-sparse search pipelines
- `src/vectordb/haystack/metadata_filtering/` - Structured field-level filtering pipelines
- `src/vectordb/haystack/mmr/` - Diversity-aware retrieval using Maximal Marginal Relevance
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
