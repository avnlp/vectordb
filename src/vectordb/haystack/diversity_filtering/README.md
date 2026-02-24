# Diversity Filtering

This module provides result diversification for Retrieval-Augmented Generation pipelines built on Haystack. It ensures that retrieved documents cover a broad range of topics rather than returning near-duplicate or narrowly focused results. By grouping retrieved documents into clusters and selecting representatives from each group, the diversity filtering step produces a result set that balances relevance to the query with coverage of distinct subtopics.

The module includes two custom Haystack components -- a clustering-based diversity ranker and a diversity metrics calculator -- alongside per-database indexing and search pipelines. An optional RAG generation step can produce answers grounded in the diversified document set.

## Overview

- Retrieves a large candidate set from the vector database, then applies diversity filtering to select a smaller, more varied subset
- Supports Maximum Margin Relevance (MMR) ranking to balance relevance and diversity via a configurable lambda parameter
- Provides clustering-based diversity ranking using KMeans or HDBSCAN to group documents by topic and select representatives
- Calculates diversity metrics including average pairwise cosine distance, cluster coverage, and diversity-relevance trade-off scores
- Integrates optional RAG generation to produce answers grounded in the diversified document set
- All behavior is parameterized through YAML configuration files with environment variable substitution

## How It Works

### Indexing

Each database has a dedicated indexing pipeline that loads documents from a configured dataset (such as TriviaQA, ARC, PopQA, FactScore, or Earnings Calls), generates dense embeddings using a sentence transformer model, and writes the embedded documents to the target vector database.

By default, indexing is **incremental** - documents are upserted into existing collections without data loss. Set `recreate: true` in the configuration to delete and recreate the collection before indexing.

### Search with Diversity Filtering

The search pipeline first retrieves a broad set of candidate documents (controlled by the retrieval top-k setting). It then applies a diversity ranker -- either the built-in Haystack sentence transformer diversity ranker using MMR, or the custom clustering-based ranker -- to select a smaller number of documents that maximize both relevance and diversity. The MMR lambda parameter controls the trade-off: values closer to zero favor diversity while values closer to one favor relevance.

### Clustering-Based Ranking

The clustering diversity ranker groups document embeddings into clusters using KMeans or HDBSCAN. It then selects the most representative document from each cluster based on a configurable selection strategy (such as choosing the document closest to each cluster centroid). This approach ensures that the final result set covers multiple distinct topic areas.

### Diversity Metrics

The diversity metrics calculator evaluates the quality of the diversified result set. It computes average pairwise cosine distance between selected documents, measures cluster coverage, and produces a combined diversity-relevance score. These metrics can be used to tune the filtering parameters.

### RAG Generation

When enabled, the pipeline passes the diversified documents to a language model (via Groq or OpenAI API) to generate an answer grounded in the retrieved context.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Qdrant | Supported | Indexing and search pipelines |
| Pinecone | Supported | Indexing and search pipelines |
| Weaviate | Supported | Indexing and search pipelines |
| Chroma | Supported | Indexing and search pipelines |
| Milvus | Supported | Indexing and search pipelines |

## Configuration

Configuration is driven by YAML files stored in the `configs/` directory, organized by database and dataset. Below is an example showing the key configuration sections:

```yaml
dataset:
  name: triviaqa
  split: test
  max_documents: 1000

embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  batch_size: 32
  device: null

index:
  name: triviaqa_diversity
  recreate: false  # Default: false (incremental/upsert). Set true to recreate collection

retrieval:
  top_k_candidates: 100

diversity:
  algorithm: maximum_margin_relevance
  top_k: 10
  mmr_lambda: 0.5
  similarity_metric: cosine

rag:
  enabled: false
  provider: groq
  model: llama-3.3-70b-versatile
  temperature: 0.7
  max_tokens: 2048

vectordb:
  type: qdrant
  qdrant:
    url: ${QDRANT_URL}
    api_key: ${QDRANT_API_KEY}
```

## Directory Structure

```
diversity_filtering/
├── __init__.py                              # Package exports
├── README.md                                # This file
├── components/                              # Custom Haystack components
│   ├── __init__.py
│   ├── clustering_diversity_ranker.py       # KMeans/HDBSCAN clustering ranker
│   └── diversity_metrics_calculator.py      # Pairwise distance and coverage metrics
├── pipelines/                               # Per-database indexing and search
│   ├── __init__.py
│   ├── chroma_indexing.py
│   ├── chroma_search.py
│   ├── milvus_indexing.py
│   ├── milvus_search.py
│   ├── pinecone_indexing.py
│   ├── pinecone_search.py
│   ├── qdrant_indexing.py
│   ├── qdrant_search.py
│   ├── weaviate_indexing.py
│   └── weaviate_search.py
├── utils/                                   # Shared utilities
│   ├── __init__.py
│   ├── config_loader.py                     # YAML config loading and validation
│   └── prompts.py                           # RAG prompt templates
└── configs/                                 # YAML configuration files
    ├── arc_diversity_config.yaml
    ├── earnings_calls_diversity_config.yaml
    ├── factscore_diversity_config.yaml
    ├── popqa_diversity_config.yaml
    ├── triviaqa_diversity_config.yaml
    └── qdrant/                              # Database-specific configs
```

## Related Modules

- `src/vectordb/haystack/dense_indexing/` - Dense embedding indexing pipelines
- `src/vectordb/haystack/rag/` - Standard RAG pipelines without diversity filtering
- `src/vectordb/dataloaders/haystack/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and Earnings Calls
