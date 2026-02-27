# Diversity Filtering

Result diversification for retrieval pipelines that ensures retrieved documents cover a broad range of topics rather than returning near-duplicate or narrowly focused results. By removing redundant documents and selecting representatives from distinct content clusters, the diversity filtering step produces a result set that balances relevance to the query with coverage of different perspectives.

This module is particularly valuable when queries could be answered from multiple angles or when the knowledge base contains overlapping content. Instead of returning five documents that all say the same thing, diversity filtering ensures the result set covers distinct aspects of the topic.

## Overview

- Removes near-duplicate documents based on embedding similarity thresholds
- Supports clustering-based diversity selection using KMeans or HDBSCAN
- Retrieves a larger candidate set, then filters down to diverse representatives
- Configurable similarity threshold for determining document redundancy
- Balances relevance and diversity through adjustable parameters
- Works with all five vector databases using a consistent interface
- Can be combined with RAG for answer generation from diverse sources
- Configuration-driven through YAML files with environment variable substitution

## How It Works

### Indexing Phase

Each database has a dedicated indexing pipeline that loads documents from a configured dataset (such as TriviaQA, ARC, PopQA, FactScore, or Earnings Calls), generates dense embeddings using a sentence transformer model, and writes the embedded documents to the target vector database. The indexing process is identical to standard semantic search indexing.

### Search with Diversity Filtering

The search pipeline first retrieves a broad set of candidate documents, typically three to five times the number ultimately requested. It then analyzes the embeddings of these candidates to identify and remove redundant documents. Documents with similarity scores above a configurable threshold are considered duplicates, and only one representative is kept from each redundant group.

For clustering-based approaches, the pipeline groups documents into topic clusters and selects the most relevant document from each cluster. This ensures the final result set covers multiple distinct topic areas rather than variations of the same content.

### Diversity Strategies

**Threshold-based filtering** compares document embeddings pairwise and removes documents that exceed a similarity threshold. The pipeline keeps the most relevant document from each similar group. This approach is fast and effective for removing obvious duplicates.

**Clustering-based selection** groups documents into clusters using algorithms like KMeans or HDBSCAN. The pipeline selects one representative from each cluster, ensuring topic diversity. The number of clusters can be configured to control the diversity-relevance trade-off.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | Namespace-scoped diversity filtering |
| Weaviate | Supported | Collection-based filtering |
| Chroma | Supported | Works with local and persistent storage |
| Milvus | Supported | Partition-aware filtering |
| Qdrant | Supported | Payload-filtered candidate retrieval |

## Configuration

Configuration is driven by YAML files stored in the `configs/` directory, organized by database and dataset. The configuration controls the candidate retrieval count, diversity method, similarity threshold, and optional RAG generation.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "diversity-index"
  namespace: ""

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32

diversity:
  method: "threshold"  # or "clustering"
  candidate_multiplier: 3  # Retrieve 3x the final count
  similarity_threshold: 0.85  # Documents above this are considered duplicates
  # For clustering method:
  # num_clusters: 5
  # samples_per_cluster: 2

search:
  top_k: 10

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  level: "INFO"
```

## Directory Structure

```
diversity_filtering/
├── __init__.py                        # Package exports
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone diversity indexing
│   ├── weaviate.py                    # Weaviate diversity indexing
│   ├── chroma.py                      # Chroma diversity indexing
│   ├── milvus.py                      # Milvus diversity indexing
│   └── qdrant.py                      # Qdrant diversity indexing
├── search/                            # Database-specific search with diversity
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone diversity search
│   ├── weaviate.py                    # Weaviate diversity search
│   ├── chroma.py                      # Chroma diversity search
│   ├── milvus.py                      # Milvus diversity search
│   └── qdrant.py                      # Qdrant diversity search
└── configs/                           # YAML configs organized by database
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── weaviate_triviaqa.yaml
    └── ...                            # (25+ config files total)
```

## Related Modules

- `src/vectordb/langchain/mmr/` - Maximal Marginal Relevance for relevance-diversity balance
- `src/vectordb/langchain/semantic_search/` - Standard semantic search without diversity
- `src/vectordb/langchain/reranking/` - Two-stage retrieval for improved relevance
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and Earnings Calls
