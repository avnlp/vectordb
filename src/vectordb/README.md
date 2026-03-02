# VectorDB

VectorDB is a Python toolkit for building, comparing, and benchmarking retrieval and Retrieval-Augmented Generation (RAG) pipelines across multiple vector databases and two popular AI frameworks: Haystack and LangChain.

## What Is This?

Modern RAG systems need to be evaluated across different vector database backends and different retrieval strategies before choosing one for production. VectorDB provides a single codebase where every combination of backend, framework, and retrieval feature can be exercised using the same datasets, configuration format, and evaluation metrics.

The toolkit is organized around three ideas:

1. **Backend wrappers** that normalize five vector databases (Pinecone, Weaviate, Chroma, Milvus, Qdrant) into a consistent interface.
2. **Feature modules** that implement retrieval patterns (semantic search, hybrid indexing, reranking, filtering, compression, and more) for both Haystack and LangChain, each in its own directory with configs, indexing, and search code.
3. **Dataloaders and evaluation** that load standard QA benchmarks, convert them to framework documents, and compute standard retrieval metrics (Recall\@k, Precision\@k, MRR, NDCG\@k, Hit Rate).

## Who This Is For

- Teams evaluating retrieval strategies and vector backends before committing to a production architecture.
- Practitioners who want to benchmark one feature improvement at a time (for example, adding reranking on top of a semantic baseline).
- Builders who need reusable RAG components that work across Haystack and LangChain without rewriting indexing and search logic from scratch.

## What You Can Build

- Baseline semantic retrieval pipelines (dense vector similarity search).
- Hybrid retrieval pipelines combining dense semantic and sparse lexical signals.
- Reranked, diversity-aware, and MMR-filtered retrieval flows.
- Metadata-filtered, namespaced, and multi-tenant retrieval systems.
- Query-enhanced retrieval with multi-query generation, HyDE, and step-back prompting.
- Contextually compressed retrieval that trims noisy context before generation.
- Cost-aware RAG with token budget controls.
- Agentic RAG with iterative self-reflection and multi-step retrieval loops.
- Parent-document retrieval that indexes small chunks but returns larger parent context.

## Installation

This project uses `uv` for dependency management. Python 3.11 or later is required.

```bash
# Install uv if you do not have it
pip install uv

# Install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

The package is installed from source as `vectordb`. All imports use `from vectordb.*` paths.

## Recommended Starting Path

1. Pick one framework (`haystack` or `langchain`) based on your existing stack.
2. Pick one backend (`pinecone`, `weaviate`, `chroma`, `milvus`, or `qdrant`) based on your deployment model.
3. Start with `semantic_search` to establish a quality and latency baseline.
4. Add one advanced feature at a time (`metadata_filtering`, `reranking`, `hybrid_indexing`, and so on).
5. Introduce `multi_tenancy` or `namespaces` once the baseline pipeline quality is stable.
6. Use `cost_optimized_rag` and `agentic_rag` for production-hardening and complex tasks.

## How to Choose a Feature

| Need | Feature |
|---|---|
| Broad natural-language matching | `semantic_search` |
| Keyword + semantic robustness | `hybrid_indexing` |
| Hard constraints (date, tenant, type) | `metadata_filtering` |
| Better final ranking | `reranking` |
| Relevance + diversity balance | `mmr` or `diversity_filtering` |
| Noisy or oversized context | `contextual_compression` |
| Underspecified queries | `query_enhancement` |
| Iterative multi-step reasoning | `agentic_rag` |
| Long docs with fragment matching | `parent_document_retrieval` |
| Structured JSON documents | `json_indexing` |
| Lexical keyword-heavy workloads | `sparse_indexing` |
| Data isolation per customer | `multi_tenancy` |
| Logical data segmentation | `namespaces` |
| Production budget controls | `cost_optimized_rag` |

## Core Modules

| Module | Purpose |
|---|---|
| `databases/` | Backend wrappers that normalize Pinecone, Weaviate, Chroma, Milvus, and Qdrant into a consistent interface for both Haystack and LangChain feature modules. |
| `dataloaders/` | Dataset normalization layer that loads TriviaQA, ARC, PopQA, FActScore, and Earnings Calls datasets into a common record format and converts them to framework-specific document objects. |
| `haystack/` | Complete Haystack implementations of all 17 retrieval feature patterns, each with per-backend configs, indexing scripts, and search scripts. |
| `langchain/` | Complete LangChain implementations of all 17 retrieval feature patterns, parallel in structure to the Haystack module. |
| `utils/` | Shared configuration loading, evaluation metrics, sparse embedding conversion, document ID management, scope injection, output structures, and logging. |

## Configuration Format

All pipelines use YAML configuration files with environment variable substitution. Variables follow `${VAR}` or `${VAR:-default}` syntax.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "my-index"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

dataloader:
  dataset: "triviaqa"
  split: "test"
  limit: 500

rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048
```

Each feature module contains per-backend, per-dataset config files under its `configs/` directory.

## Supported Backends

| Backend | Client Type | Best For |
|---|---|---|
| Pinecone | Managed cloud (GRPC) | Namespace-based multi-tenancy, serverless scale |
| Weaviate | Cloud or self-hosted | Hybrid + generative search, native BM25 |
| Chroma | Local or HTTP | Development, prototyping, lightweight local setups |
| Milvus / Zilliz | Self-hosted or managed | Partition-key multi-tenancy, scalable infrastructure |
| Qdrant | Self-hosted or cloud | Named vectors, quantization, payload filtering |

## Supported Datasets

| Dataset | Type | Records |
|---|---|---|
| TriviaQA (`trivia_qa`) | Open-domain QA with evidence | ~500 index, ~100 eval queries |
| ARC (`ai2_arc`) | Science QA | ~1000 index, ~200 eval queries |
| PopQA (`akariasai/PopQA`) | Entity-centric QA | ~500 index, ~100 eval queries |
| FActScore (`dskar/FActScore`) | Factuality-focused QA | ~500 index, ~100 eval queries |
| Earnings Calls (`lamini/earnings-calls-qa`) | Financial QA transcripts | ~300 index, ~50 eval queries |

## Evaluation Metrics

The `utils/evaluation.py` module computes these metrics over retrieved document IDs compared to ground-truth relevant IDs:

- **Recall\@k**: Fraction of relevant documents retrieved in the top-k results.
- **Precision\@k**: Fraction of top-k results that are relevant.
- **MRR (Mean Reciprocal Rank)**: Average of the reciprocal rank of the first relevant document.
- **NDCG\@k**: Rank-aware metric that normalizes by the ideal ranking.
- **Hit Rate**: Binary indicator of whether any relevant document appeared in the top-k.

## LLM Integration

Generation and agentic components use the Groq API with Llama models via OpenAI-compatible endpoints. Set the `GROQ_API_KEY` environment variable to enable generation, query enhancement, agentic routing, and context compression features.

## Project Layout

```
src/vectordb/
├── databases/          # Backend wrappers (Chroma, Milvus, Pinecone, Qdrant, Weaviate)
├── dataloaders/        # Dataset loading, normalization, and evaluation query extraction
│   └── datasets/       # Per-dataset loaders (TriviaQA, ARC, PopQA, FActScore, Earnings Calls)
├── haystack/           # Haystack feature implementations
│   ├── agentic_rag/
│   ├── components/     # Reusable Haystack components (router, compressor, enhancer)
│   ├── contextual_compression/
│   ├── cost_optimized_rag/
│   ├── diversity_filtering/
│   ├── hybrid_indexing/
│   ├── json_indexing/
│   ├── metadata_filtering/
│   ├── mmr/
│   ├── multi_tenancy/
│   ├── namespaces/
│   ├── parent_document_retrieval/
│   ├── query_enhancement/
│   ├── reranking/
│   ├── semantic_search/
│   ├── sparse_indexing/
│   └── utils/          # Haystack-specific helpers (embeddings, reranker, fusion, RAG)
├── langchain/          # LangChain feature implementations (parallel structure to haystack/)
│   ├── components/     # Reusable LangChain components
│   └── utils/          # LangChain-specific helpers
└── utils/              # Shared utilities (config, evaluation, sparse, ids, scope, logging)
```

## License

MIT License. See the `LICENSE` file in the repository root for details.
