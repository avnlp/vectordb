# Maximal Marginal Relevance

Maximal Marginal Relevance (MMR) is a diversity-aware reranking strategy that reduces redundancy in search results. Standard nearest-neighbor retrieval often returns clusters of nearly identical documents, especially when the corpus contains paraphrased or overlapping content. MMR addresses this by selecting documents that are both relevant to the query and dissimilar to the documents already selected, producing a result set that covers a broader range of information.

The module implements MMR as a post-retrieval step. The pipeline first over-fetches candidate documents from the vector database using standard dense retrieval, then applies the MMR algorithm to rerank and filter those candidates down to the requested number of results. An optional RAG step can generate a natural-language answer from the diverse result set.

## Overview

- Diversity-aware reranking that balances query relevance against inter-document similarity
- Lambda parameter to control the relevance-diversity tradeoff (1.0 = pure relevance, 0.0 = pure diversity)
- Over-fetching strategy that retrieves more candidates than needed, then reranks with MMR
- Optional retrieval-augmented generation using an LLM to synthesize answers from diverse results
- Configuration-driven through YAML files with environment variable substitution
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Indexing Phase

The indexing pipeline is identical to standard dense indexing. Documents are loaded from a dataset, embedded using a sentence-transformer model, and stored in the target vector database. No special indexing structure is required for MMR, because the reranking happens entirely at search time.

### Search Phase

The search pipeline performs three steps. First, it embeds the query and retrieves an over-sized set of candidate documents from the database (for example, 50 candidates when only 10 final results are needed). Second, it applies the MMR algorithm to iteratively select documents. The algorithm scores each remaining candidate using the formula:

```
MMR(d) = lambda * similarity(query, d) - (1 - lambda) * max(similarity(d, d_selected))
```

At each step, the candidate with the highest MMR score is added to the selected set. A lambda value close to 1.0 prioritizes relevance, a value close to 0.0 prioritizes diversity, and 0.5 provides an even balance. Third, if RAG is enabled, the selected documents are passed as context to an LLM to generate a grounded answer.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | Standard dense retrieval with MMR reranking |
| Weaviate | Supported | Standard dense retrieval with MMR reranking |
| Chroma | Supported | Standard dense retrieval with MMR reranking |
| Milvus | Supported | Standard dense retrieval with MMR reranking |
| Qdrant | Supported | Standard dense retrieval with MMR reranking |

## Configuration

Each pipeline is driven by a YAML configuration file. Below is an example showing the key sections:

```yaml
dataloader:
  type: "triviaqa"         # triviaqa, arc, popqa, factscore, earnings_calls
  split: "test"
  limit: 1000

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

mmr:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 10                # Final number of results after reranking
  top_k_candidates: 50     # Number of candidates to over-fetch
  lambda_threshold: 0.5    # Relevance-diversity tradeoff

qdrant:                    # Database-specific section (one per config)
  url: "${QDRANT_URL}"
  api_key: "${QDRANT_API_KEY}"
  collection_name: "mmr_triviaqa"
  dimension: 384
  metric: "Cosine"

search:
  top_k: 10

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
mmr/
├── __init__.py                        # Package exports for all pipelines
├── README.md
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone dense indexing for MMR
│   ├── weaviate.py                    # Weaviate dense indexing for MMR
│   ├── chroma.py                      # Chroma dense indexing for MMR
│   ├── milvus.py                      # Milvus dense indexing for MMR
│   └── qdrant.py                      # Qdrant dense indexing for MMR
├── search/                            # Database-specific MMR search pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone retrieval with MMR reranking
│   ├── weaviate.py                    # Weaviate retrieval with MMR reranking
│   ├── chroma.py                      # Chroma retrieval with MMR reranking
│   ├── milvus.py                      # Milvus retrieval with MMR reranking
│   └── qdrant.py                      # Qdrant retrieval with MMR reranking
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

- `src/vectordb/haystack/semantic_search/` - Standard dense semantic search (without diversity reranking)
- `src/vectordb/haystack/hybrid_indexing/` - Hybrid dense-plus-sparse search pipelines
- `src/vectordb/haystack/utils/` - Shared utilities for configuration, embedding, data loading, and reranking
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
