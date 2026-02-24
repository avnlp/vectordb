# Query Enhancement

Query expansion pipelines that use LLM-generated query variations to improve retrieval recall. Instead of relying on a single query, the pipeline generates multiple reformulations of the user's question and executes each one independently against the vector database. The results from all queries are then merged using reciprocal rank fusion to produce a single, unified ranking that captures relevant documents that any individual query might have missed.

Three enhancement strategies are available. Multi-query generates alternative phrasings of the original question to capture different aspects of the intent. Hypothetical document embedding generates synthetic answer passages and uses them as search queries, bridging the gap between question-style queries and answer-style documents. Step-back prompting generates broader contextual questions that retrieve background information useful for answering the original question.

## Overview

- Three query enhancement strategies: multi-query, hypothetical document embedding, and step-back prompting
- Parallel execution of all generated queries for low-latency search
- N-way reciprocal rank fusion to merge results from multiple query variations
- Content-based deduplication to eliminate redundant documents across result sets
- Optional RAG answer generation from the fused retrieval results
- Configuration-driven strategy selection and parameter tuning via YAML

## How It Works

### Indexing

The indexing pipeline loads documents from a configured dataset, generates dense embeddings using a sentence transformer model, and upserts the embedded documents into the target vector database collection. The indexing process is identical across all enhancement strategies since query expansion only affects the search stage.

### Search

The search pipeline first generates query variations based on the configured strategy. For multi-query, the LLM produces alternative phrasings. For hypothetical document embedding, the LLM generates synthetic passages that might answer the question. For step-back prompting, the LLM generates a broader version of the question. Each generated query (plus the original) is embedded and searched against the vector database in parallel using a thread pool. The per-query result lists are combined using reciprocal rank fusion with a configurable constant (default k=60), then deduplicated by content hash. If RAG generation is enabled, the fused documents are passed to an LLM to produce a final answer.

## Supported Databases

| Database | Indexing Module | Search Module | Notes |
|----------|----------------|---------------|-------|
| Pinecone | `indexing/pinecone.py` | `search/pinecone.py` | Serverless managed service |
| Weaviate | `indexing/weaviate.py` | `search/weaviate.py` | GraphQL-based queries |
| Chroma | `indexing/chroma.py` | `search/chroma.py` | Local or cloud deployment |
| Milvus | `indexing/milvus.py` | `search/milvus.py` | Distributed vector database |
| Qdrant | `indexing/qdrant.py` | `search/qdrant.py` | Payload filtering support |

## Configuration

Each database has per-dataset YAML configuration files organized in subdirectories. The configuration specifies the enhancement strategy type, LLM model for query generation, fusion parameters, and optional RAG generation settings.

```yaml
dataloader:
  type: "triviaqa"
  params:
    limit: 1000

embeddings:
  model: "all-MiniLM-L6-v2"

query_enhancement:
  type: "hyde"                  # "multi_query", "hyde", or "step_back"
  num_queries: 3                # Number of alternative queries (multi-query)
  num_hyde_docs: 3              # Number of hypothetical documents (hyde)
  llm:
    model: "llama-3.3-70b-versatile"
    api_key: "${GROQ_API_KEY}"
  fusion_method: "rrf"
  rrf_k: 60
  top_k: 10

qdrant:
  url: "${QDRANT_URL}"
  api_key: "${QDRANT_API_KEY}"
  collection_name: "triviaqa-query-enhancement"

logging:
  level: "INFO"
  name: "qdrant_query_enhancement"

rag:
  enabled: true
```

## Directory Structure

```
src/vectordb/haystack/query_enhancement/
├── __init__.py                        # Module docstring and public API
├── configs/                           # YAML configs organized by database
│   ├── pinecone/                      # Pinecone configs (5 dataset files)
│   │   ├── triviaqa.yaml
│   │   ├── arc.yaml
│   │   ├── popqa.yaml
│   │   ├── factscore.yaml
│   │   └── earnings_calls.yaml
│   ├── qdrant/                        # Qdrant configs (5 dataset files)
│   ├── milvus/                        # Milvus configs (5 dataset files)
│   ├── chroma/                        # Chroma configs (5 dataset files)
│   └── weaviate/                      # Weaviate configs (5 dataset files)
├── indexing/                          # Indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone document indexing
│   ├── qdrant.py                      # Qdrant document indexing
│   ├── milvus.py                      # Milvus document indexing
│   ├── chroma.py                      # Chroma document indexing
│   └── weaviate.py                    # Weaviate document indexing
├── search/                            # Search pipelines with query expansion
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone multi-query search
│   ├── qdrant.py                      # Qdrant multi-query search
│   ├── milvus.py                      # Milvus multi-query search
│   ├── chroma.py                      # Chroma multi-query search
│   └── weaviate.py                    # Weaviate multi-query search
├── utils/                             # Shared utilities
│   ├── __init__.py
│   ├── config.py                      # Configuration loading and validation
│   ├── dataloader.py                  # Dataset loading helpers
│   ├── embeddings.py                  # Embedder initialization
│   ├── fusion.py                      # Reciprocal rank fusion and deduplication
│   ├── llm.py                         # LLM generator initialization
│   └── types.py                       # Type definitions
└── README.md
```

## Related Modules

- `src/vectordb/haystack/components/query_enhancer.py` - Query enhancer component used by search pipelines
- `src/vectordb/haystack/reranking/` - Two-stage retrieval with reranking (complementary approach)
- `src/vectordb/haystack/semantic_search/` - Standard dense indexing without query expansion
- `src/vectordb/haystack/agentic_rag/` - Full RAG pipelines with generation
