# Contextual Compression

Post-retrieval document compression pipelines that reduce the volume of retrieved content before passing it to a downstream language model. The pipeline retrieves more documents than ultimately needed, then applies a compression step that either filters or summarizes the candidate set down to the most relevant passages. This reduces token consumption during generation while preserving the information most pertinent to the query.

Two compression strategies are available. Reranking uses cross-encoder models to score each retrieved document against the query and retains only the highest-scoring results. LLM extraction uses a language model to read each retrieved document and extract only the passages relevant to the query, producing condensed versions that carry the same information in fewer tokens.

## Overview

- Two compression strategies: cross-encoder reranking and LLM-based passage extraction
- Over-fetches documents during initial retrieval to ensure sufficient candidates for compression
- Reranking operates without additional LLM API calls for cost-efficient filtering
- LLM extraction produces condensed document summaries focused on query-relevant content
- Pipeline classes for each supported vector database following a consistent pattern
- Token counting utilities to measure compression ratios and token savings
- Configuration-driven through YAML files with environment variable substitution

## How It Works

### Indexing Phase

The indexing pipeline loads documents from a configured dataset, generates dense embeddings using a sentence transformer model, and writes the embedded documents to the target vector database collection. Each database has a dedicated indexing implementation that handles collection creation and document upsert according to its specific API. The indexing process is identical regardless of which compression strategy will be used at search time.

### Search Phase

The search pipeline embeds the incoming query and retrieves an initial candidate set from the vector database, typically fetching more documents than ultimately requested. The candidates are then passed through the configured compressor. When using reranking, a cross-encoder model scores each query-document pair and the results are sorted by relevance score, keeping only the top results. When using LLM extraction, a language model processes each document and extracts only the passages relevant to the query, returning condensed versions. The compressed result set is truncated to the final requested count and returned.

### Compression Strategies

**Reranking** uses a cross-encoder model that processes query-document pairs together to produce a relevance score. The pipeline retrieves candidates using vector similarity, then reranks them using the cross-encoder to identify the most relevant documents. This approach is fast and does not require additional LLM calls.

**LLM Extraction** uses a language model to read each retrieved document and extract only the sentences or passages relevant to the query. This can significantly reduce token count but adds latency due to the LLM processing step. The extracted passages are concatenated to form a compressed context.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | Uses namespaces for logical partitioning |
| Weaviate | Supported | Uses collections for organization |
| Chroma | Supported | Lightweight local or client-server deployment |
| Milvus | Supported | Uses collections with configurable metrics |
| Qdrant | Supported | Supports both local and server deployments |

## Configuration

Each database has per-dataset YAML configuration files organized by database and dataset, with separate files for each compression strategy. The configuration controls database connection parameters, embedding model selection, retrieval top-k for over-fetching, compression type and model, and optional RAG generation.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "compressed-index"
  namespace: ""

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  batch_size: 32

compression:
  mode: "reranking"  # or "llm_extraction"
  reranker:
    type: "cross_encoder"
    model: "BAAI/bge-reranker-v2-m3"
    top_k: 5
  llm_extraction:
    model: "llama-3.3-70b-versatile"
    api_key: "${GROQ_API_KEY}"
    max_tokens_per_doc: 200

rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  level: "INFO"
```

## Directory Structure

```
contextual_compression/
├── __init__.py                        # Package exports
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone compression indexing
│   ├── weaviate.py                    # Weaviate compression indexing
│   ├── chroma.py                      # Chroma compression indexing
│   ├── milvus.py                      # Milvus compression indexing
│   └── qdrant.py                      # Qdrant compression indexing
├── search/                            # Database-specific search with compression
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone compression search
│   ├── weaviate.py                    # Weaviate compression search
│   ├── chroma.py                      # Chroma compression search
│   ├── milvus.py                      # Milvus compression search
│   └── qdrant.py                      # Qdrant compression search
└── configs/                           # YAML configs organized by database
    ├── pinecone_triviaqa_reranking.yaml
    ├── pinecone_triviaqa_llm.yaml
    ├── weaviate_arc_reranking.yaml
    └── ...                            # (25+ config files total)
```

## Related Modules

- `src/vectordb/langchain/reranking/` - Dedicated two-stage reranking pipelines
- `src/vectordb/langchain/semantic_search/` - Standard semantic search without compression
- `src/vectordb/langchain/cost_optimized_rag/` - Cost-aware RAG with optional compression
- `src/vectordb/langchain/components/` - Reusable components including the ContextCompressor
