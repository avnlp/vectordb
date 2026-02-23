# Reranking

Two-stage retrieval pipelines that combine fast dense vector search with cross-encoder reranking for improved relevance. The first stage retrieves a large candidate set using approximate nearest neighbor search over document embeddings. The second stage rescores each candidate using a cross-encoder model that jointly processes the query and document text, producing more accurate relevance judgments than embedding similarity alone.

The initial retrieval over-fetches by a factor of three to ensure the reranker has a broad candidate pool. After rescoring, only the top results are returned, balancing precision with latency. All five supported vector databases share a consistent interface through a common base class pattern, with database-specific logic isolated in thin subclasses.

## Overview

- Two-stage retrieval: fast dense search followed by cross-encoder reranking
- Over-fetches candidates by 3x in the first stage to maximize reranker effectiveness
- Supports multiple cross-encoder models including multilingual and lightweight options
- Separate indexing and search pipelines for independent scaling
- Configuration-driven via YAML files with environment variable support
- Integrated evaluation using contextual recall, precision, and faithfulness metrics

## How It Works

### Indexing

The indexing pipeline loads documents from a configured dataset, generates dense embeddings using a sentence transformer model, creates or recreates the target collection in the vector database, and upserts all embedded documents. Each database has a dedicated indexing implementation that handles connection setup and collection management according to its specific API.

### Search

The search pipeline embeds the incoming query using the same sentence transformer model used during indexing. It then retrieves three times the requested number of results from the vector database using dense similarity search. The over-fetched candidate set is passed to a cross-encoder reranker, which jointly encodes each query-document pair to produce a more accurate relevance score. The reranked results are truncated to the final requested count and returned.

## Supported Databases

| Database | Indexing Module | Search Module | Notes |
|----------|----------------|---------------|-------|
| Pinecone | `indexing/pinecone.py` | `search/pinecone.py` | Serverless managed service |
| Weaviate | `indexing/weaviate.py` | `search/weaviate.py` | GraphQL-based queries |
| Chroma | `indexing/chroma.py` | `search/chroma.py` | Local or cloud deployment |
| Milvus | `indexing/milvus.py` | `search/milvus.py` | Distributed vector database |
| Qdrant | `indexing/qdrant.py` | `search/qdrant.py` | Payload filtering support |

## Configuration

Each database-dataset combination has its own YAML configuration file. The configuration controls database connection parameters, embedding model selection, reranker model and top-k settings, dataloader type and limits, and evaluation metrics.

```yaml
qdrant:
  host: "${QDRANT_HOST:-localhost}"
  port: "${QDRANT_PORT:-6333}"
  api_key: "${QDRANT_API_KEY:-}"

collection:
  name: "triviaqa_reranking"

dataloader:
  type: "triviaqa"
  dataset_name: "trivia_qa"
  split: "test"
  limit: null

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  batch_size: 32

reranker:
  type: "cross_encoder"
  model: "BAAI/bge-reranker-v2-m3"
  top_k: 5

evaluation:
  enabled: true
  metrics:
    - contextual_recall
    - contextual_precision

logging:
  name: "qdrant_reranking"
  level: "INFO"
```

## Directory Structure

```
src/vectordb/haystack/reranking/
├── __init__.py                        # Public exports for all pipeline classes
├── configs/                           # YAML configs (25 files: 5 databases x 5 datasets)
│   ├── qdrant_triviaqa.yaml
│   ├── qdrant_arc.yaml
│   ├── qdrant_popqa.yaml
│   ├── qdrant_factscore.yaml
│   ├── qdrant_earnings_calls.yaml
│   ├── pinecone_triviaqa.yaml
│   ├── ...
│   └── chroma_earnings_calls.yaml
├── indexing/                          # Indexing pipelines
│   ├── __init__.py
│   ├── chroma.py                      # Chroma indexing implementation
│   ├── milvus.py                      # Milvus indexing implementation
│   ├── pinecone.py                    # Pinecone indexing implementation
│   ├── qdrant.py                      # Qdrant indexing implementation
│   └── weaviate.py                    # Weaviate indexing implementation
├── search/                            # Search pipelines with reranking
│   ├── __init__.py
│   ├── chroma.py                      # Chroma search with reranking
│   ├── milvus.py                      # Milvus search with reranking
│   ├── pinecone.py                    # Pinecone search with reranking
│   ├── qdrant.py                      # Qdrant search with reranking
│   └── weaviate.py                    # Weaviate search with reranking
└── README.md
```

## Related Modules

- `src/vectordb/haystack/utils/` - Shared configuration loading, embedder factory, and reranker factory
- `src/vectordb/haystack/contextual_compression/` - Alternative post-retrieval compression approach
- `src/vectordb/haystack/semantic_search/` - Dense indexing without reranking stage
- `src/vectordb/haystack/agentic_rag/` - Full RAG pipelines with generation
