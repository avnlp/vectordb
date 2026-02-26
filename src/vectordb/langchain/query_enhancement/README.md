# Query Enhancement

Query rewriting and expansion techniques that improve retrieval quality by generating variations of the original query. These techniques address the vocabulary mismatch problem where user queries use different terminology than the indexed documents, and help retrieve comprehensive results for underspecified or ambiguous queries.

The module implements three established query enhancement strategies: multi-query generation, hypothetical document embeddings (HyDE), and step-back prompting. Each technique addresses different types of retrieval challenges and can be used individually or combined for maximum effectiveness.

## Overview

- Multi-query generation creates variations of the original query to capture different phrasings
- HyDE generates a hypothetical answer document to bridge vocabulary gaps
- Step-back prompting abstracts the query to retrieve broader context
- Query fusion combines results from multiple enhanced queries
- All techniques work with any embedding model and vector database
- Configuration-driven through YAML files with environment variable substitution
- Minimal latency overhead when using efficient LLM providers

## How It Works

### Multi-Query Enhancement

The multi-query strategy generates multiple variations of the original query, each phrased differently but semantically equivalent. Each variation is used to retrieve documents, and the results are merged using fusion algorithms like Reciprocal Rank Fusion (RRF) or weighted averaging.

For example, a query "What is photosynthesis?" might generate variations like:
- "Explain the process of photosynthesis"
- "How do plants convert sunlight to energy?"
- "Describe photosynthesis in plants"

Each variation retrieves different documents, and the fusion step combines them into a comprehensive result set that captures documents using different terminology.

### HyDE (Hypothetical Document Embeddings)

HyDE addresses the vocabulary mismatch problem by asking a language model to write a hypothetical answer to the query. This hypothetical answer uses the same terminology as the knowledge base, bridging the gap between query vocabulary and document vocabulary. The hypothetical document is embedded and used for retrieval instead of the original query.

For example, a query "Why is the sky blue?" might generate a hypothetical answer: "The sky appears blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight are scattered in all directions by gas molecules in Earth's atmosphere." This answer contains technical terms like "Rayleigh scattering" that match the terminology in scientific documents.

### Step-Back Prompting

Step-back prompting abstracts the query to a more general level to retrieve broader context before answering specific questions. This technique is useful when the specific query might miss relevant background information needed for a complete answer.

For example, a query "What is the capital of France?" might first generate a step-back query: "Tell me about France, including its geography and major cities." The broader context retrieved helps ground the specific answer with relevant background information.

### Query Fusion

When multiple enhanced queries are generated, the pipeline retrieves documents for each and combines the results. Two fusion strategies are supported:

**Reciprocal Rank Fusion (RRF)** scores documents based on their rank in each result list and combines scores without requiring score normalization. Documents that appear high in multiple result lists receive the best combined scores.

**Weighted Average Fusion** combines similarity scores from each query, with weights reflecting confidence in each enhanced query. This requires score normalization but can produce more nuanced rankings.

## Supported Databases

All five vector databases are fully supported. Query enhancement is applied before the retrieval step and works with any database backend.

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | All enhancement strategies work with Pinecone |
| Weaviate | Supported | All enhancement strategies work with Weaviate |
| Chroma | Supported | All enhancement strategies work with Chroma |
| Milvus | Supported | All enhancement strategies work with Milvus |
| Qdrant | Supported | All enhancement strategies work with Qdrant |

## Configuration

Configuration is stored in YAML files organized by database and dataset. The configuration specifies which enhancement strategies to enable, how many variations to generate, and fusion settings.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "enhanced-index"
  namespace: ""

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32

query_enhancement:
  strategies:
    - "multi_query"
    - "hyde"
    # - "step_back"  # Uncomment to enable

  multi_query:
    enabled: true
    num_variations: 3
    fusion_method: "rrf"  # or "weighted_average"
    rrf_k: 60  # RRF constant

  hyde:
    enabled: true
    model: "llama-3.3-70b-versatile"
    api_key: "${GROQ_API_KEY}"
    temperature: 0.7
    max_tokens: 256

  step_back:
    enabled: false
    model: "llama-3.3-70b-versatile"
    api_key: "${GROQ_API_KEY}"
    temperature: 0.5

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
query_enhancement/
├── __init__.py                        # Package exports
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone enhanced indexing
│   ├── weaviate.py                    # Weaviate enhanced indexing
│   ├── chroma.py                      # Chroma enhanced indexing
│   ├── milvus.py                      # Milvus enhanced indexing
│   └── qdrant.py                      # Qdrant enhanced indexing
├── search/                            # Database-specific enhanced search
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone enhanced search
│   ├── weaviate.py                    # Weaviate enhanced search
│   ├── chroma.py                      # Chroma enhanced search
│   ├── milvus.py                      # Milvus enhanced search
│   └── qdrant.py                      # Qdrant enhanced search
└── configs/                           # YAML configs organized by database
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── weaviate_triviaqa.yaml
    └── ...                            # (25+ config files total)
```

## Related Modules

- `src/vectordb/langchain/semantic_search/` - Standard search without query enhancement
- `src/vectordb/langchain/hybrid_indexing/` - Hybrid search with query enhancement support
- `src/vectordb/langchain/components/` - QueryEnhancer component implementation
- `src/vectordb/langchain/utils/` - Fusion and diversification utilities
