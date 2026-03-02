# Configuration Reference

## 1. What This Feature Is

This document provides a **complete inventory of all configuration keys** used across VectorDB pipelines. All configs use YAML format with environment variable substitution.

## 2. Configuration Format

### Environment Variable Syntax

| Syntax | Behavior | Example |
|--------|----------|---------|
| **`${VAR}`** | Substitute with env value, empty if unset | `${PINECONE_API_KEY}` |
| **`${VAR:-default}`** | Substitute with VAR if set, else default | `${DEVICE:-cpu}` |

### Loading Configs

```python
from vectordb.utils import load_config

# Load from file
config = load_config("config.yaml")

# Config is a dict with resolved env vars
api_key = config["pinecone"]["api_key"]
```

## 3. Core Configuration Keys

### Embeddings (Required for all features)

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Required: model path or alias
  device: "cpu"                                     # Optional: "cpu" or "cuda"
  batch_size: 32                                    # Optional: batch size for embedding
  trust_remote_code: false                          # Optional: for custom models
  dimension: 384                                    # Optional: embedding dimension
```

**Model Aliases**:

- `qwen3` → `Qwen/Qwen3-Embedding-0.6B`
- `minilm` → `sentence-transformers/all-MiniLM-L6-v2`
- `mpnet` → `sentence-transformers/all-mpnet-base-v2`

### Sparse Embeddings (For hybrid/sparse features)

```yaml
sparse:
  model: "naver/splade-cocondenser-ensembledistil"  # Required: SPLADE model
  max_length: 512                                    # Optional: max sequence length
```

### Dataloader (For indexing)

```yaml
dataloader:
  type: "triviaqa"          # Required: dataset type
  dataset_name: "trivia_qa" # Optional: HuggingFace dataset name
  config: "rc"              # Optional: dataset config
  split: "test"             # Optional: dataset split (default: "test")
  limit: 500                # Optional: record limit
  params:                   # Optional: additional params
    dataset_name: "trivia_qa"
```

### Search (For search pipelines)

```yaml
search:
  top_k: 10              # Optional: default result count (default: 10)
  reranking_enabled: true  # Optional: enable reranking (default: false)
  hybrid_enabled: false    # Optional: enable hybrid search (not always used)
  metadata_filtering_enabled: false  # Optional: enable filtering (not always used)
```

### RAG (Optional for generation)

```yaml
rag:
  enabled: true                      # Optional: enable RAG (default: false)
  model: "llama-3.3-70b-versatile"   # Optional: LLM model
  api_key: "${GROQ_API_KEY}"         # Required if enabled
  api_base_url: "https://api.groq.com/openai/v1"  # Optional: API endpoint
  temperature: 0.7                   # Optional: generation temperature
  max_tokens: 2048                   # Optional: max output tokens
  prompt_template: "..."             # Optional: custom prompt
```

### Logging

```yaml
logging:
  level: "INFO"    # Optional: log level (default: "INFO")
  name: "pipeline" # Optional: logger name
```

## 4. Backend Configuration Keys

### Chroma

```yaml
chroma:
  collection_name: "my-collection"  # Required: collection name
  persist_dir: "./chroma"           # Optional: persistent storage path
  host: "localhost"                 # Optional: server host (for HTTP client)
  port: 8000                        # Optional: server port
  api_key: "${CHROMA_API_KEY}"      # Optional: API key for Chroma Cloud
  tenant: "default_tenant"          # Optional: tenant name
  database: "default_database"      # Optional: database name
  dimension: 384                    # Optional: embedding dimension
  batch_size: 100                   # Optional: batch size for upsert
  fusion_strategy: "rrf"            # Optional: "rrf" or "linear"
  is_persistent: true               # Optional: use persistent client
```

### Milvus

```yaml
milvus:
  uri: "http://localhost:19530"     # Required: server URI
  token: ""                         # Optional: Zilliz Cloud token
  collection_name: "my-collection"  # Required: collection name
  dimension: 384                    # Optional: embedding dimension
  batch_size: 100                   # Optional: batch size
  recreate: false                   # Optional: recreate collection
  use_partition_key: false          # Optional: enable partition key
  partition_key_field: "tenant_id"  # Optional: partition key field name
  ranker_type: "rrf"                # Optional: "rrf" or "weighted"
  indexing:
    partitions:
      enabled: false                # Optional: enable partitions
    payload_indexes:                # Optional: payload indexes
      - field_name: "category"
        field_schema: "keyword"
```

### Pinecone

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"    # Required: API key
  index_name: "my-index"            # Required: index name
  namespace: ""                     # Optional: namespace for multi-tenancy
  dimension: 384                    # Optional: embedding dimension
  metric: "cosine"                  # Optional: "cosine", "euclidean", "dotproduct"
  cloud: "aws"                      # Optional: cloud provider
  region: "us-east-1"               # Optional: region
  batch_size: 100                   # Optional: batch size
  show_progress: true               # Optional: show progress during upsert
  alpha: 0.5                        # Optional: hybrid alpha weight
  recreate: false                   # Optional: recreate index
```

### Qdrant

```yaml
qdrant:
  url: "http://localhost:6333"      # Required: server URL
  api_key: "${QDRANT_API_KEY}"      # Optional: API key for Qdrant Cloud
  collection_name: "my-collection"  # Required: collection name
  dimension: 384                    # Optional: embedding dimension
  batch_size: 100                   # Optional: batch size
  recreate: false                   # Optional: recreate collection
  path: null                        # Optional: local storage path
  dense_vector_name: "dense"        # Optional: dense vector name
  sparse_vector_name: "sparse"      # Optional: sparse vector name
  quantization:                     # Optional: quantization config
    type: "scalar"                  # "scalar" or "binary"
    quantile: 0.99
    always_ram: true
  payload_indexes:                  # Optional: payload indexes
    - field_name: "tenant_id"
      field_schema: "keyword"
      is_tenant: true
```

### Weaviate

```yaml
weaviate:
  cluster_url: "https://xxx.weaviate.cloud"  # Required: cluster URL
  api_key: "${WEAVIATE_API_KEY}"             # Required: API key
  collection_name: "MyCollection"            # Required: collection name (PascalCase)
  dimension: 384                             # Optional: embedding dimension
  batch_size: 100                            # Optional: batch size
  recreate: false                            # Optional: recreate collection
  headers:                                   # Optional: additional headers
    X-OpenAI-Api-Key: "${OPENAI_API_KEY}"
  alpha: 0.5                                 # Optional: hybrid alpha weight
  enable_multi_tenancy: false                # Optional: enable multi-tenancy
```

## 5. Feature-Specific Configuration Keys

### MMR

```yaml
mmr:
  lambda_param: 0.5        # Optional: relevance vs diversity (0-1)
  top_k: 10                # Optional: final result count
  top_k_candidates: 50     # Optional: candidate pool size
```

### Reranking

```yaml
reranking:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Optional: reranker model
  top_k: 10                 # Optional: final result count
  top_k_candidates: 30      # Optional: candidate pool size
  device: "cpu"             # Optional: device
  batch_size: 32            # Optional: batch size
```

### Query Enhancement

```yaml
query_enhancement:
  type: "multi_query"       # Required: "multi_query", "hyde", "step_back", "rewriting"
  num_queries: 5            # Optional: for multi_query
  num_hyde_docs: 3          # Optional: for hyde
  llm:
    model: "llama-3.3-70b-versatile"  # Required: LLM model
    api_key: "${GROQ_API_KEY}"        # Required: API key
    api_base_url: "https://api.groq.com/openai/v1"
    temperature: 0.7                  # Optional: temperature
    max_tokens: 256                   # Optional: max tokens
```

### Contextual Compression

```yaml
compression:
  type: "llm"               # Required: "llm" or "embeddings"
  llm:
    model: "llama-3.3-70b-versatile"  # Required for LLM
    api_key: "${GROQ_API_KEY}"
    temperature: 0
    max_tokens: 2048
  embeddings:
    similarity_threshold: 0.7  # Required for embeddings
    k: 10
```

### Chunking (Parent Document Retrieval)

```yaml
chunking:
  parent_chunk_size: 2000   # Optional: parent document size
  child_chunk_size: 200     # Optional: child chunk size
  overlap: 50               # Optional: overlap between chunks
```

### Retrieval (Parent Document Retrieval / Contextual Compression)

```yaml
retrieval:
  top_k: 10                 # Optional: child chunks to retrieve
  parent_top_k: 5           # Optional: parent documents to return
  top_k_candidates: 20      # Optional: pre-compression candidate count
```

### Diversity Filtering

```yaml
diversity:
  top_k: 10                 # Optional: final result count
  algorithm: "maximum_margin_relevance"  # Optional: "maximum_margin_relevance", "greedy_diversity_order", "clustering"
  similarity_metric: "cosine"  # Optional: "cosine" or "dot_product"
  mmr_lambda: 0.5           # Optional: MMR lambda (0-1)
  diversity_threshold: 0.7  # Optional: similarity threshold
  max_similar_docs: 3       # Optional: max similar docs to keep
```

### Agentic RAG

```yaml
agentic_rag:
  model: "llama-3.3-70b-versatile"  # Required: router model
  api_key: "${GROQ_API_KEY}"        # Required: API key
  routing_enabled: true             # Optional: enable routing
  self_reflection_enabled: true     # Optional: enable reflection
  max_iterations: 2                 # Optional: max refinement iterations
  quality_threshold: 75             # Optional: quality threshold (0-100)
  reflection_context_top_k: 3       # Optional: context docs for reflection
  fallback_tool: "generate"         # Optional: fallback tool
  max_retries: 3                    # Optional: max retries
  retry_delay_seconds: 1            # Optional: retry delay
```

### Indexing

```yaml
indexing:
  batch_size: 32            # Optional: batch size for indexing
  quantization:             # Optional: quantization config (not always used)
    type: "scalar"
  partitions:               # Optional: partition config (Milvus)
    enabled: false
  payload_indexes:          # Optional: payload indexes
    - field_name: "category"
      field_schema: "keyword"
```

### Multi-Tenancy

```yaml
tenant:
  id: "acme-prod"           # Optional: tenant ID (or TENANT_ID env)
  name: "Acme Production"   # Optional: tenant name
  metadata:                 # Optional: tenant metadata
    department: "engineering"
    tier: "enterprise"

multitenancy:
  strategy: "namespace"     # Optional: isolation strategy
  field_name: "tenant_id"   # Optional: tenant field name
  auto_create_tenant: true  # Optional: auto-create tenant
  partition_key_isolation: true  # Optional: use partition key
  num_partitions: 10        # Optional: number of partitions
```

### Namespaces

```yaml
namespaces:
  - namespace_id: "tenant-acme"
    isolation_strategy: "partition"
    backend_config:
      partition_key: "tenant_id"

namespace:
  default: "default-namespace"  # Optional: default namespace
```

### Metadata Filtering

```yaml
metadata_filtering:
  schema:
    allowed_fields: ["category", "year", "source", "title"]
    allowed_operators: ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"]
  test_query: "What is the capital of France?"  # Optional: default query
  test_filters:                                 # Optional: default filters
    - field: "category"
      operator: "$eq"
      value: "science"
    - field: "year"
      operator: "$gte"
      value: 2020
```

### Cost-Optimized RAG

```yaml
collection:
  name: "rag-demo"          # Optional: collection/index name

generator:
  enabled: true             # Optional: enable generation
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  max_tokens: 2048
  temperature: 0.7

reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 10
```

## 6. Default Dataset Limits

```python
from vectordb.utils.config import get_dataset_limits

limits = get_dataset_limits("triviaqa")
# Returns: {"index_limit": 500, "eval_limit": 100}
```

| Dataset | Index Limit | Eval Limit |
|---------|-------------|------------|
| **triviaqa** | 500 | 100 |
| **arc** | 1000 | 200 |
| **popqa** | 500 | 100 |
| **factscore** | 500 | 100 |
| **earnings_calls** | 300 | 50 |

## 7. Source Walkthrough Map

### Config Loading

| File | Purpose |
|------|---------|
| `src/vectordb/utils/config.py` | Config loading, env var resolution |
| `src/vectordb/utils/config_loader.py` | ConfigLoader class |
| `src/vectordb/haystack/utils/config.py` | Haystack-specific config |
| `src/vectordb/langchain/utils/config.py` | LangChain-specific config |

### Default Limits

| File | Purpose |
|------|---------|
| `src/vectordb/utils/config.py` | `DATASET_LIMITS` dictionary |

### Feature Configs

| Directory | Purpose |
|-----------|---------|
| `src/vectordb/haystack/*/configs/` | Haystack feature configs |
| `src/vectordb/langchain/*/configs/` | LangChain feature configs |

---

**Related Documentation**:

- **Public API Reference** (`docs/reference/public-api.md`): Complete API inventory
- **Core Shared Utils** (`docs/core/shared-utils.md`): Configuration utilities
