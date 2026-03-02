# VectorDB Documentation

**Comprehensive documentation for building, comparing, and benchmarking retrieval and RAG pipelines across multiple vector databases and AI frameworks.**

## Quick Start

```bash
# Install dependencies
uv sync
source .venv/bin/activate

# Run a simple semantic search
from vectordb.haystack.semantic_search import ChromaSemanticSearchPipeline

pipeline = ChromaSemanticSearchPipeline("config.yaml")
results = pipeline.search("What is RAG?", top_k=5)
```

## How to Navigate

### Start Here If You're New

1. **[Core: VectorDB](core/vectordb.md)** - Package architecture and module boundaries
2. **[Framework Overview](#frameworks)** - Choose Haystack or LangChain
3. **[Feature Catalog](#feature-catalog)** - Pick your retrieval feature
4. **[Configuration Reference](reference/config-reference.md)** - All config keys

### Find What You Need

| I want to... | Start with |
|--------------|------------|
| **Understand the architecture** | [Core: VectorDB](core/vectordb.md) |
| **Choose a framework** | [Haystack Overview](haystack/overview.md) or [LangChain Overview](langchain/overview.md) |
| **Implement semantic search** | [Semantic Search](haystack/semantic-search.md) |
| **Add metadata filtering** | [Metadata Filtering](haystack/metadata-filtering.md) |
| **Improve retrieval quality** | [Reranking](haystack/reranking.md) or [Hybrid Indexing](haystack/hybrid-indexing.md) |
| **Reduce token costs** | [Contextual Compression](haystack/contextual-compression.md) |
| **Support multi-tenancy** | [Multi-Tenancy](haystack/multi-tenancy.md) or [Namespaces](haystack/namespaces.md) |
| **Configure pipelines** | [Configuration Reference](reference/config-reference.md) |

## Core

Fundamental building blocks used across all features.

| Document | Description |
|----------|-------------|
| **[Core: VectorDB](core/vectordb.md)** | Package architecture, module boundaries, backend support matrix |
| **[Core: Databases](core/databases.md)** | Backend wrappers (Chroma, Milvus, Pinecone, Qdrant, Weaviate) |
| **[Core: Dataloaders](core/dataloaders.md)** | Dataset loading, normalization, evaluation query extraction |
| **[Core: Shared Utils](core/shared-utils.md)** | Config loading, evaluation metrics, document conversion, sparse embeddings |

## Frameworks

Choose your framework based on your existing stack and preferences.

### Haystack

| Document | Description |
|----------|-------------|
| **[Haystack Overview](haystack/overview.md)** | Feature catalog, architecture, decision guidance |
| **[Semantic Search](haystack/semantic-search.md)** | Dense vector similarity search |
| **[Hybrid Indexing](haystack/hybrid-indexing.md)** | Dense + sparse retrieval with fusion |
| **[Sparse Indexing](haystack/sparse-indexing.md)** | Lexical/keyword matching with SPLADE/BM25 |
| **[Reranking](haystack/reranking.md)** | Cross-encoder second-stage scoring |
| **[MMR](haystack/mmr.md)** | Maximal Marginal Relevance for diversity |
| **[Metadata Filtering](haystack/metadata-filtering.md)** | Structured constraints on retrieval |
| **[Query Enhancement](haystack/query-enhancement.md)** | Multi-query, HyDE, step-back expansion |
| **[Parent Document Retrieval](haystack/parent-document-retrieval.md)** | Chunk indexing with parent context return |
| **[Contextual Compression](haystack/contextual-compression.md)** | Context trimming before generation |
| **[Contextual Compression Indexing](haystack/contextual-compression-indexing.md)** | Compression at index time |
| **[Cost-Optimized RAG](haystack/cost-optimized-rag.md)** | Budget-aware retrieval and generation |
| **[Diversity Filtering](haystack/diversity-filtering.md)** | Post-retrieval redundancy reduction |
| **[Agentic RAG](haystack/agentic-rag.md)** | Tool routing and self-reflection loops |
| **[Multi-Tenancy](haystack/multi-tenancy.md)** | Tenant-isolated indexing and retrieval |
| **[Namespaces](haystack/namespaces.md)** | Logical data partitioning |
| **[JSON Indexing](haystack/json-indexing.md)** | JSON-aware schema and filtering |
| **[Components](haystack/components.md)** | Reusable advanced-RAG components |
| **[Utils](haystack/utils.md)** | Shared framework utilities |

### LangChain

| Document | Description |
|----------|-------------|
| **[LangChain Overview](langchain/overview.md)** | Feature catalog, architecture, comparison with Haystack |
| **[Semantic Search](langchain/semantic-search.md)** | Dense vector similarity search |
| **[Hybrid Indexing](langchain/hybrid-indexing.md)** | Dense + sparse retrieval with fusion |
| **[Sparse Indexing](langchain/sparse-indexing.md)** | Lexical/keyword matching with SPLADE/BM25 |
| **[Reranking](langchain/reranking.md)** | Cross-encoder second-stage scoring |
| **[MMR](langchain/mmr.md)** | Maximal Marginal Relevance for diversity |
| **[Metadata Filtering](langchain/metadata-filtering.md)** | Structured constraints on retrieval |
| **[Query Enhancement](langchain/query-enhancement.md)** | Multi-query, HyDE, step-back expansion |
| **[Parent Document Retrieval](langchain/parent-document-retrieval.md)** | Chunk indexing with parent context return |
| **[Contextual Compression](langchain/contextual-compression.md)** | Context trimming before generation |
| **[Cost-Optimized RAG](langchain/cost-optimized-rag.md)** | Budget-aware retrieval and generation |
| **[Diversity Filtering](langchain/diversity-filtering.md)** | Post-retrieval redundancy reduction |
| **[Agentic RAG](langchain/agentic-rag.md)** | Tool routing and self-reflection loops |
| **[Namespaces](langchain/namespaces.md)** | Logical data partitioning |
| **[Components](langchain/components.md)** | Reusable LangChain components |
| **[Utils](langchain/utils.md)** | Shared framework utilities |

## Reference

Authoritative reference for APIs and configuration.

| Document | Description |
|----------|-------------|
| **[Public API Reference](reference/public-api.md)** | Complete inventory of all exported classes and functions |
| **[Configuration Reference](reference/config-reference.md)** | All configuration keys with defaults and examples |

## Feature Catalog

Quick reference for choosing retrieval features.

### Retrieval Core

| Feature | When to Use | Frameworks |
|---------|-------------|------------|
| **[Semantic Search](haystack/semantic-search.md)** | Baseline retrieval, conceptual matching | Haystack, LangChain |
| **[Hybrid Indexing](haystack/hybrid-indexing.md)** | Best of both semantic and keyword | Haystack, LangChain |
| **[Sparse Indexing](haystack/sparse-indexing.md)** | Exact term matching, IDs, acronyms | Haystack, LangChain |

### Ranking & Diversity

| Feature | When to Use | Frameworks |
|---------|-------------|------------|
| **[Reranking](haystack/reranking.md)** | High precision needed | Haystack, LangChain |
| **[MMR](haystack/mmr.md)** | Avoid redundant results | Haystack, LangChain |
| **[Diversity Filtering](haystack/diversity-filtering.md)** | Multi-faceted queries | Haystack, LangChain |

### Query/Context Transformation

| Feature | When to Use | Frameworks |
|---------|-------------|------------|
| **[Query Enhancement](haystack/query-enhancement.md)** | Ambiguous or sparse queries | Haystack, LangChain |
| **[Contextual Compression](haystack/contextual-compression.md)** | Token budget limits | Haystack, LangChain |
| **[Agentic RAG](haystack/agentic-rag.md)** | Complex multi-step queries | Haystack, LangChain |

### Data Shaping & Isolation

| Feature | When to Use | Frameworks |
|---------|-------------|------------|
| **[Metadata Filtering](haystack/metadata-filtering.md)** | Domain/time/source filtering | Haystack, LangChain |
| **[JSON Indexing](haystack/json-indexing.md)** | Structured JSON documents | Haystack, LangChain |
| **[Parent Document Retrieval](haystack/parent-document-retrieval.md)** | Long documents with chunks | Haystack, LangChain |
| **[Namespaces](haystack/namespaces.md)** | Environment separation | Haystack, LangChain |
| **[Multi-Tenancy](haystack/multi-tenancy.md)** | Multi-customer SaaS | Haystack, LangChain |

### Cost/Governance

| Feature | When to Use | Frameworks |
|---------|-------------|------------|
| **[Cost-Optimized RAG](haystack/cost-optimized-rag.md)** | Budget constraints | Haystack, LangChain |

## Backend Support Matrix

| Feature | Pinecone | Weaviate | Chroma | Milvus | Qdrant |
|---------|----------|----------|--------|--------|--------|
| **Semantic Search** | Yes | Yes | Yes | Yes | Yes |
| **Hybrid Search** | Yes | Yes | Yes | Yes | Yes |
| **Sparse Indexing** | Yes | Yes (BM25) | Partial | Yes | Yes |
| **Metadata Filtering** | Yes | Yes | Yes | Yes | Yes |
| **MMR** | Yes | Yes | Yes | Yes | Yes |
| **Reranking** | Yes | Yes | Yes | Yes | Yes |
| **Multi-Tenancy** | Yes | Yes | Yes | Yes | Yes |
| **Namespaces** | Yes | Yes | Yes | Yes | Yes |

## Recommended Onboarding Path

1. **Start with semantic search** on your target backend to establish a baseline
2. **Extract evaluation queries** from your dataset and measure baseline metrics
3. **Add one improvement feature at a time**:
   - Start with **reranking** (usually highest single-step gain)
   - Or **hybrid indexing** (for mixed query types)
4. **Once quality is stable**, layer in **multi-tenancy** or **namespaces** for data isolation
5. **Use cost-optimized RAG** to find acceptable quality-cost tradeoffs
6. **Use agentic RAG** for complex multi-step reasoning tasks

## Installation

```bash
# Install uv if you don't have it
pip install uv

# Install all dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## Quick Example

```python
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.semantic_search import ChromaSemanticSearchPipeline

# Load dataset
loader = DataloaderCatalog.create("triviaqa", split="test", limit=500)
dataset = loader.load()

# Index documents
indexer = ChromaSemanticIndexingPipeline("config.yaml")
indexer.run(documents=dataset.to_haystack())

# Search
searcher = ChromaSemanticSearchPipeline("config.yaml")
results = searcher.search("What is retrieval augmented generation?", top_k=5)

for doc in results["documents"]:
    print(f"Score {doc.score}: {doc.content[:100]}")
```

## Getting Help

- **Architecture questions**: See [Core: VectorDB](core/vectordb.md)
- **Feature selection**: See [Feature Catalog](#feature-catalog)
- **Configuration issues**: See [Configuration Reference](reference/config-reference.md)
- **API reference**: See [Public API Reference](reference/public-api.md)
