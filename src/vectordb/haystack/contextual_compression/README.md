# Contextual Compression

Post-retrieval document compression pipelines that reduce the volume of retrieved content before passing it to a downstream language model. The pipeline retrieves more documents than ultimately needed (over-fetching by 2x), then applies a compression step that either filters or summarizes the candidate set down to the most relevant passages. This reduces token consumption during generation while preserving the information most pertinent to the query.

Two compression strategies are available. Cross-encoder reranking scores each retrieved document against the query and retains only the highest-scoring results, discarding irrelevant candidates without any LLM calls. LLM-based extraction uses a language model to read each retrieved document and extract only the passages relevant to the query, producing condensed versions that carry the same information in fewer tokens.

## Overview

- Two compression strategies: cross-encoder reranking and LLM-based passage extraction
- Over-fetches by 2x during initial retrieval to ensure the compressor has sufficient candidates
- Cross-encoder reranking operates locally without LLM API calls for cost-efficient filtering
- LLM extraction produces condensed document summaries focused on query-relevant content
- Shared base class with database-specific subclasses handling only connection and retrieval logic
- Token counting utilities to measure compression ratios and token savings

## How It Works

### Indexing

The indexing pipeline loads documents from a configured dataset, generates dense embeddings using a sentence transformer model, and writes the embedded documents to the target vector database collection. Each database has a dedicated indexing implementation that handles collection creation and document upsert according to its specific API. The indexing process is identical regardless of which compression strategy will be used at search time.

### Search

The search pipeline embeds the incoming query and retrieves an initial candidate set from the vector database, typically fetching twice the number of documents ultimately requested. The candidates are then passed through the configured compressor. When using reranking, a cross-encoder model scores each query-document pair and the results are sorted by relevance score, keeping only the top results. When using LLM extraction, a language model processes each document and extracts only the passages relevant to the query, returning condensed versions. The compressed result set is truncated to the final requested count and returned.

## Supported Databases

| Database | Search Module | Indexing Module | Notes |
|----------|---------------|-----------------|-------|
| Pinecone | `search/pinecone_search.py` | `indexing/pinecone_indexing.py` | Serverless managed service |
| Weaviate | `search/weaviate_search.py` | `indexing/weaviate_indexing.py` | GraphQL-based queries |
| Chroma | `search/chroma_search.py` | `indexing/chroma_indexing.py` | Local or cloud deployment |
| Milvus | `search/milvus_search.py` | `indexing/milvus_indexing.py` | Distributed vector database |
| Qdrant | `search/qdrant_search.py` | `indexing/qdrant_indexing.py` | Payload filtering support |

## Configuration

Each database has per-dataset YAML configuration files organized by database and dataset, with separate files for each compression strategy. The configuration controls database connection parameters, embedding model selection, retrieval top-k for over-fetching, compression type and model, and optional RAG generation.

```yaml
milvus:
  host: "${MILVUS_HOST:-localhost}"
  port: "${MILVUS_PORT:-19530}"

collection:
  name: "triviaqa_compression"

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  dimension: 1024

retrieval:
  top_k: 20

compression:
  type: "reranking"
  top_k: 5
  reranker:
    type: "cross_encoder"
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

rag:
  enabled: false
  llm:
    provider: "groq"
    model: "llama-3.3-70b-versatile"

logging:
  name: "contextual_compression"
  level: "INFO"
```

## Directory Structure

```
src/vectordb/haystack/contextual_compression/
├── __init__.py                        # Public exports for pipeline and utility classes
├── base.py                            # Abstract base class with shared orchestration logic
├── compression_utils.py               # Compressor factory, token counter, result formatting
├── evaluation.py                      # Compression quality metrics calculation
├── configs/                           # YAML configs (50 files: 5 DBs x 5 datasets x 2 strategies)
│   ├── __init__.py
│   ├── milvus/
│   │   ├── triviaqa/
│   │   │   ├── reranking.yaml
│   │   │   └── llm_extraction.yaml
│   │   ├── arc/
│   │   ├── popqa/
│   │   ├── factscore/
│   │   └── earnings_calls/
│   ├── pinecone/                      # Same nested structure per dataset
│   ├── qdrant/
│   ├── chroma/
│   └── weaviate/
├── indexing/                          # Database-specific indexing implementations
│   ├── __init__.py
│   ├── base_indexing.py               # Base indexing class
│   ├── milvus_indexing.py             # Milvus document indexing
│   ├── pinecone_indexing.py           # Pinecone document indexing
│   ├── qdrant_indexing.py             # Qdrant document indexing
│   ├── chroma_indexing.py             # Chroma document indexing
│   └── weaviate_indexing.py           # Weaviate document indexing
├── search/                            # Database-specific search with compression
│   ├── __init__.py
│   ├── milvus_search.py               # Milvus retrieval and compression
│   ├── pinecone_search.py             # Pinecone retrieval and compression
│   ├── qdrant_search.py               # Qdrant retrieval and compression
│   ├── chroma_search.py               # Chroma retrieval and compression
│   └── weaviate_search.py             # Weaviate retrieval and compression
└── README.md
```

## Related Modules

- `src/vectordb/haystack/reranking/` - Dedicated two-stage reranking pipelines (related but separate focus)
- `src/vectordb/haystack/semantic_search/` - Standard dense indexing without compression
- `src/vectordb/haystack/cost_optimized_rag/` - Cost-aware RAG with conditional compression
- `src/vectordb/haystack/agentic_rag/` - Full RAG pipelines with generation
