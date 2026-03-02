# Haystack: JSON Indexing

## 1. What This Feature Is

JSON indexing provides a backend-portable pattern for storing and retrieving Haystack `Document` objects where:

- **`Document.content`**: Embedded for semantic similarity search
- **`Document.meta`**: Retained for structured metadata filtering

This module implements five explicit backend pairs (not one generic runtime class):

| Backend | Indexer | Searcher |
|---------|---------|----------|
| **Milvus** | `MilvusJSONIndexer` | `MilvusJSONSearcher` |
| **Qdrant** | `QdrantJSONIndexer` | `QdrantJSONSearcher` |
| **Pinecone** | `PineconeJSONIndexer` | `PineconeJSONSearcher` |
| **Weaviate** | `WeaviateJSONIndexer` | `WeaviateJSONSearcher` |
| **Chroma** | `ChromaJSONIndexer` | `ChromaJSONSearcher` |

All are exported from `vectordb.haystack.json_indexing`.

## 2. Why It Exists in Retrieval/RAG

RAG retrieval often needs **two constraints simultaneously**:

1. **Semantic relevance**: Vector similarity over document content
2. **Structured eligibility**: Metadata constraints (category, tenant, time, status)

This module codifies that combination so each vector backend can be driven with the same high-level Python interface:

- `indexer.run()` for indexing
- `searcher.search(query, filters, top_k)` for retrieval

### Design Alignment

This approach aligns with official platform behavior:

- **Haystack**: Separates document embedding (`SentenceTransformersDocumentEmbedder`) from query embedding (`SentenceTransformersTextEmbedder`)
- **Vector DBs**: Expose backend-specific metadata filtering constructs:
  - Milvus: Boolean expression strings
  - Qdrant: `Filter` objects with `FieldCondition`
  - Weaviate: `Filter` class with property-based API
  - Pinecone/Chroma: MongoDB-style dict syntax

## 3. Indexing Pipeline: Step-by-Step

```mermaid
flowchart TD
    A[Config Loading] --> B[Logger + DB Connection]
    B --> C[Initialize Document Embedder]
    C --> D[Warm Up Embedder]
    D --> E[Create Dataloader]
    E --> F[Load Documents to Haystack]
    F --> G{Create Collection?}
    G -->|Yes| H[Backend-Specific Create]
    G -->|No| I[Skip Creation]
    H --> J[Embed Documents]
    I --> J
    J --> K[Write with Metadata]
    K --> L[Return documents_indexed]
```

### Common Indexing Sequence

All indexers follow this high-level sequence:

1. **Load config**: `load_config(config_or_path)` with env var resolution
2. **Build logger**: From `logging.name` and `logging.level`
3. **Connect DB wrapper**: Backend-specific connection
4. **Initialize embedder**: `create_document_embedder()` + `warm_up()`
5. **Create dataloader**: `DataloaderCatalog.create(...)`
6. **Load documents**: `loader.load().to_haystack()`
7. **Log limit**: Optionally log configured limit
8. **Create collection/index**: Backend-specific method
9. **Embed documents**: `embed_documents(documents)`
10. **Write documents**: Backend wrapper method
11. **Return**: `{"documents_indexed": len(documents)}`

### Backend-Specific Indexing

| Backend | Collection Creation | Write Method | Special Handling |
|---------|---------------------|--------------|------------------|
| **Milvus** | `create_collection(collection_name, dimension, use_sparse=False)` | `insert_documents()` | Probes embedding dimension first |
| **Qdrant** | `create_collection(dimension)` | `index_documents()` | Injects `collection.name` into config |
| **Pinecone** | Delegated to wrapper | `upsert_documents(namespace=...)` | No explicit index creation |
| **Weaviate** | `create_collection(collection_name, skip_vectorization=False)` | `upsert_documents()` | Requires `cluster_url` + `api_key` |
| **Chroma** | `create_collection(collection_name=...)` | `upsert_documents()` | Wrapper-config driven |

## 4. Search Pipeline: Step-by-Step

```mermaid
flowchart TD
    A[Config Loading] --> B[Logger + DB Connection]
    B --> C[Initialize Text Embedder]
    C --> D[Warm Up Embedder]
    D --> E[Resolve top_k]
    E --> F[Embed Query Text]
    F --> G[Build Backend Filter]
    G --> H[Execute Backend Search]
    H --> I[Return Results]
    I --> J{Normalize?}
    J -->|Yes| K[normalize_search_results]
    J -->|No| L[Raw Wrapper Results]
```

### Common Search Sequence

All searchers follow this sequence:

1. **Load config**: `load_config(config_or_path)`
2. **Build logger**: From `logging.name` and `logging.level`
3. **Connect DB wrapper**: Backend-specific connection
4. **Initialize embedder**: `create_text_embedder()` + `warm_up()`
5. **Resolve top_k**: Explicit arg → `search.top_k` → default `10`
6. **Embed query**: `text_embedder.run(text=query)` → `result["embedding"]`
7. **Convert filters**: Generic dict → backend-native filter form
8. **Execute search**: Backend wrapper search method
9. **Return**: Wrapper results directly (no automatic normalization)

### Backend-Specific Search

| Backend | Search Method | Filter Builder | Filter Type |
|---------|---------------|----------------|-------------|
| **Milvus** | `vector_db.search(..., filter_expr=...)` | `build_milvus_filter(filters, json_field_name)` | Expression string |
| **Qdrant** | `vector_db.search(..., query_filter=...)` | `build_qdrant_filter(filters)` | `Filter` object |
| **Pinecone** | `vector_db.search(..., filter=...)` | `build_pinecone_filter(filters)` | Dict |
| **Weaviate** | `vector_db.hybrid_search(..., where=...)` | `build_weaviate_filter(filters)` | `Filter` object |
| **Chroma** | `vector_db.search(..., where=...)` | `build_chroma_filter(filters)` | `where` dict |

## 5. When to Use It

Use JSON indexing when **all** of the following are true:

- **You want semantic retrieval over free text (`Document.content`)**
- **You need precise metadata filtering from `Document.meta`**
- **You need the same indexing/search contract across multiple backends**

### Typical Use Cases

| Use Case | Example |
|----------|---------|
| **QA corpora** | Source/type/topic metadata filtering |
| **Operational constraints** | Tenant, domain, language, date bands |
| **Multi-environment** | Same code path for local vs managed DB |
| **Structured + unstructured** | Content search + metadata filters |

## 6. When Not to Use It

Avoid this module when:

- **Pure semantic retrieval**: You only need vector similarity, no filters
- **Complex boolean logic**: Nested OR trees not supported by filter builders
- **Strict cross-backend equivalence**: Operator support differs across backends
- **Volatile metadata schema**: Filter key stability is low

### Operator Support Gaps

| Operator | Milvus | Qdrant | Weaviate | Pinecone | Chroma |
|----------|--------|--------|----------|----------|--------|
| `$eq` | Yes | Yes | Yes | Yes | Yes |
| `$ne` | Yes | Yes (must_not) | Yes | Yes | Yes |
| `$gt`, `$gte` | Yes | Yes | Yes | Yes | Yes |
| `$lt`, `$lte` | Yes | Yes | Yes | Yes | Yes |
| `$in`, `$nin` | No | No | No | Yes | Yes |
| `$contains` | Yes | No | No | No | No |

**Note**: Spec declares `$in/$nin` as supported, but Milvus/Qdrant/Weaviate translators don't implement them.

## 7. What This Codebase Provides

### Concrete Runtime Classes

```python
from vectordb.haystack.json_indexing import (
    # Indexers
    "MilvusJSONIndexer",
    "QdrantJSONIndexer",
    "PineconeJSONIndexer",
    "WeaviateJSONIndexer",
    "ChromaJSONIndexer",

    # Searchers
    "MilvusJSONSearcher",
    "QdrantJSONSearcher",
    "PineconeJSONSearcher",
    "WeaviateJSONSearcher",
    "ChromaJSONSearcher",
)
```

### Shared Utilities

```python
from vectordb.haystack.json_indexing.common.config import (
    load_config,  # Dict or YAML path with env resolution
)
from vectordb.haystack.json_indexing.common.embeddings import (
    create_document_embedder,  # Document embedder
    create_text_embedder,     # Query embedder
    embed_documents,          # Embed document list
    get_embedding_dimension,  # Probe embedding length
)
from vectordb.haystack.json_indexing.common.metadata import (
    flatten_metadata,  # Keep primitives, stringify non-primitives
)
from vectordb.haystack.json_indexing.common.results import (
    normalize_search_results,  # Map to {id, score, content, metadata}
)
```

### Filter Builders

```python
from vectordb.haystack.json_indexing.common.filters.milvus import (
    build_milvus_filter,
)
from vectordb.haystack.json_indexing.common.filters.qdrant import (
    build_qdrant_filter,
)
from vectordb.haystack.json_indexing.common.filters.pinecone import (
    build_pinecone_filter,
)
from vectordb.haystack.json_indexing.common.filters.weaviate import (
    build_weaviate_filter,
)
from vectordb.haystack.json_indexing.common.filters.chroma import (
    build_chroma_filter,
)
```

### Operator Specification

```python
SUPPORTED_OPERATORS = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}

def validate_filter_operator(op: str) -> None:
    """Validate filter operator against supported set."""
```

## 8. Backend-Specific Behavior Differences

### Connection and Collection Bootstrap

| Backend | Connection Fields | Collection Creation | Config Mutation |
|---------|-------------------|---------------------|-----------------|
| **Milvus** | `milvus.uri`, `milvus.token` | Explicit `create_collection()` | None |
| **Qdrant** | `qdrant.url`, `qdrant.api_key` | Explicit `create_collection()` | Injects `collection.name` |
| **Pinecone** | Wrapper-config driven | Delegated to wrapper | None |
| **Weaviate** | `weaviate.cluster_url`, `weaviate.api_key` | Explicit `create_collection()` | None |
| **Chroma** | Wrapper-config driven | Explicit `create_collection()` | None |

### Search Method Differences

| Backend | Method Used | Collection Arg | Filter Arg |
|---------|-------------|----------------|------------|
| **Milvus** | `search()` | `collection_name` | `filter_expr` (string) |
| **Qdrant** | `search()` | `collection_name` | `query_filter` (Filter object) |
| **Pinecone** | `search()` | None (from wrapper) | `filter` (dict) |
| **Weaviate** | `hybrid_search()` | `collection_name` | `where` (Filter object) |
| **Chroma** | `search()` | `collection_name` | `where` (dict) |

### Filter Translation Details

**Milvus**:

- Builds string expressions with JSON path syntax
- Supports `$contains` → `json_contains(metadata["field"], value)`
- Example: `'metadata["category"] == "tech" && metadata["year"] > 2020'`

**Qdrant**:

- Builds typed `Filter(must, must_not)` objects
- `$ne` mapped to `must_not` clause
- Mixed clauses split between `must` and `must_not`

**Weaviate**:

- Builds composable `Filter.by_property(...).<op>(...)` objects
- Combined via `&` operator
- Unsupported operators skipped (may yield weaker filters)

**Pinecone**:

- Passes operator dicts through
- Wraps scalars as `$eq`
- Supports full MongoDB-style syntax

**Chroma**:

- Emits single condition or `{"$and": [...]}` for multiple conditions
- MongoDB-style dict syntax

## 9. Configuration Semantics

### Config Input Formats

```python
# Python dict
config = {
    "embeddings": {"model": "qwen3"},
    "search": {"top_k": 10},
}

# YAML file path
config = "src/vectordb/haystack/json_indexing/configs/milvus/triviaqa.yaml"

# Both work
indexer = MilvusJSONIndexer(config)
```

### Environment Variable Substitution

| Syntax | Behavior | Example |
|--------|----------|---------|
| `${VAR}` | Env value or empty string | `${PINECONE_API_KEY}` |
| `${VAR:-default}` | Env value if set, else default | `${QDRANT_URL:-http://localhost:6333}` |

Applied recursively to nested dict/list values.

### Keys Consumed by Runtime

```yaml
# Logging
logging:
  name: "json-indexing-pipeline"
  level: "INFO"

# Embeddings
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Or "qwen3", "minilm"

# Search
search:
  top_k: 10

# Collection
collection:
  name: "json_indexed"  # Default fallback

# Dataloader
dataloader:
  type: "triviaqa"
  split: "test"
  limit: 500
  dataset_name: "trivia_qa"

# Backend-specific
milvus:
  uri: "http://localhost:19530"
  token: ""

qdrant:
  url: "http://localhost:6333"
  api_key: null

pinecone:
  api_key: "${PINECONE_API_KEY}"
  index: "my-index"
  namespace: "tenant-1"  # Optional

weaviate:
  cluster_url: "https://xxx.weaviate.cloud"
  api_key: "xxx"

chroma:
  persist_dir: "./chroma"

# Milvus-specific
indexing:
  json_field_name: "metadata"  # For JSON path prefix
```

### Defaults That Affect Behavior

| Setting | Default | Impact |
|---------|---------|--------|
| `embeddings.model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding quality/size |
| `model aliases` | `qwen3` → `Qwen/Qwen3-Embedding-0.6B` | Shortcut resolution |
| `top_k` | `10` | Search result count |
| `collection.name` | `json_indexed` | Collection identifier |

## 10. Failure Modes and Edge Cases

### Configuration Issues

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Empty YAML returns None** | `yaml.safe_load()` on empty file | Validate config before use |
| **Missing env vars** | `${VAR}` resolves to empty string | Use `${VAR:-default}` syntax |
| **Invalid credentials** | Empty API key from missing env | Validate credentials explicitly |

### Operator Support Mismatches

| Issue | Backend | Mitigation |
|-------|---------|------------|
| **`$in/$nin` not implemented** | Milvus, Qdrant, Weaviate | Use `$eq` with OR logic manually |
| **`$contains` not in spec** | Milvus only | Document as Milvus-specific feature |
| **Unsupported operators skipped** | Weaviate | Log warnings; verify filter strength |

### Filter Translation Edge Cases

| Case | Behavior | Workaround |
|------|----------|------------|
| **`$ne` in Qdrant** | Mapped to `must_not` clause | Correct for inequality; splits mixed clauses |
| **Nested OR logic** | Not supported | Flatten to AND conditions where possible |
| **Complex boolean trees** | Varies by backend | Test per backend; document limitations |

### Result Handling

| Issue | Cause | Mitigation |
|-------|-------|------------|
| **Raw wrapper results** | Searchers don't normalize automatically | Call `normalize_search_results()` explicitly |
| **Heterogeneous output shapes** | Different backends return different formats | Use normalization utility |
| **Limit not enforced post-load** | `dataloader.limit` passed to catalog | Indexers log but don't slice |

### Qdrant-Specific Issues

| Issue | Cause | Mitigation |
|-------|-------|------------|
| **Config mutation** | Indexer injects `collection.name` into `qdrant.collection_name` | Searcher doesn't mutate; be aware of difference |
| **Must/must_not splitting** | Mixed clauses split between must and must_not | Understand Filter semantics |

## 11. Practical Usage Examples

### Example 1: Milvus Indexing and Search

```python
from vectordb.haystack.json_indexing import MilvusJSONIndexer, MilvusJSONSearcher

config_path = "src/vectordb/haystack/json_indexing/configs/milvus/triviaqa.yaml"

# Index documents
index_stats = MilvusJSONIndexer(config_path).run()
print(f"Indexed {index_stats['documents_indexed']} documents")

# Search with metadata filters
results = MilvusJSONSearcher(config_path).search(
    query="Who discovered penicillin?",
    filters={"category": "science", "year": {"$gte": 1900}},
    top_k=5,
)

for doc in results:
    print(f"Score {doc['score']}: {doc['content'][:100]}")
```

### Example 2: Qdrant with Dict Config

```python
from vectordb.haystack.json_indexing import QdrantJSONIndexer, QdrantJSONSearcher

cfg = {
    "qdrant": {
        "url": "${QDRANT_URL:-http://localhost:6333}",
        "api_key": "${QDRANT_API_KEY:-}",
    },
    "collection": {"name": "triviaqa_json_indexed"},
    "dataloader": {"type": "triviaqa", "split": "test", "limit": 100},
    "embeddings": {"model": "qwen3"},
    "search": {"top_k": 10},
}

# Index
QdrantJSONIndexer(cfg).run()

# Search with inequality filter
results = QdrantJSONSearcher(cfg).search(
    query="neural network basics",
    filters={"category": "AI", "difficulty": {"$ne": "hard"}},
)
```

### Example 3: Normalize Results for Consistent Shape

```python
from vectordb.haystack.json_indexing.common.results import normalize_search_results

# Raw results vary by backend
raw_results = [
    {"id": "d1", "score": 0.91, "content": "...", "metadata": {"source": "x"}},
    {"id": "d2", "score": 0.85, "content": "...", "metadata": {"source": "y"}},
]

# Normalize to consistent shape
normalized = normalize_search_results(raw_results)
# All results have: id, score, content, metadata
```

### Example 4: Pinecone with Namespace

```python
from vectordb.haystack.json_indexing import PineconeJSONIndexer, PineconeJSONSearcher

config = {
    "pinecone": {
        "api_key": "${PINECONE_API_KEY}",
        "index": "my-index",
        "namespace": "tenant-1",  # Multi-tenancy via namespace
    },
    "embeddings": {"model": "minilm"},
    "dataloader": {"type": "triviaqa", "limit": 500},
}

# Index to specific namespace
PineconeJSONIndexer(config).run()

# Search within namespace
results = PineconeJSONSearcher(config).search(
    query="machine learning",
    filters={"source": "wikipedia"},
)
```

### Example 5: Weaviate with Complex Filters

```python
from vectordb.haystack.json_indexing import WeaviateJSONIndexer, WeaviateJSONSearcher

config_path = "src/vectordb/haystack/json_indexing/configs/weaviate/earnings_calls.yaml"

# Index
WeaviateJSONIndexer(config_path).run()

# Search with multiple conditions (AND logic)
results = WeaviateJSONSearcher(config_path).search(
    query="revenue growth",
    filters={
        "$and": [
            {"company": {"$eq": "Apple"}},
            {"quarter": {"$eq": "Q4"}},
            {"year": {"$gte": 2023}},
        ]
    },
    top_k=10,
)
```

### Example 6: Chroma with Local Persistence

```python
from vectordb.haystack.json_indexing import ChromaJSONIndexer, ChromaJSONSearcher

config = {
    "chroma": {
        "persist_dir": "./chroma-data",
    },
    "collection": {"name": "my-docs"},
    "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "dataloader": {"type": "arc", "split": "test", "limit": 200},
}

# Index to persistent storage
ChromaJSONIndexer(config).run()

# Search later (data persists)
results = ChromaJSONSearcher(config).search(
    query="photosynthesis process",
    filters={"subject": "biology"},
)
```

## 12. Source Walkthrough Map

### Public API Surface

| File | Purpose |
|------|---------|
| `src/vectordb/haystack/json_indexing/__init__.py` | Main module exports |
| `src/vectordb/haystack/json_indexing/indexing/__init__.py` | Indexer exports |
| `src/vectordb/haystack/json_indexing/search/__init__.py` | Searcher exports |

### Indexing Implementations

| File | Backend |
|------|---------|
| `indexing/milvus.py` | Milvus |
| `indexing/qdrant.py` | Qdrant |
| `indexing/pinecone.py` | Pinecone |
| `indexing/weaviate.py` | Weaviate |
| `indexing/chroma.py` | Chroma |

### Search Implementations

| File | Backend |
|------|---------|
| `search/milvus.py` | Milvus |
| `search/qdrant.py` | Qdrant |
| `search/pinecone.py` | Pinecone |
| `search/weaviate.py` | Weaviate |
| `search/chroma.py` | Chroma |

### Shared Runtime Utilities

| File | Purpose |
|------|---------|
| `common/config.py` | Config loading with env resolution |
| `common/embeddings.py` | Embedder creation and embedding utilities |
| `common/metadata.py` | Metadata flattening utilities |
| `common/results.py` | Result normalization |

### Filter Builders

| File | Purpose |
|------|---------|
| `common/filters/spec.py` | Operator specification and validation |
| `common/filters/milvus.py` | Milvus expression builder |
| `common/filters/qdrant.py` | Qdrant Filter object builder |
| `common/filters/pinecone.py` | Pinecone dict filter builder |
| `common/filters/weaviate.py` | Weaviate Filter object builder |
| `common/filters/chroma.py` | Chroma where clause builder |

### Configuration Examples

| File | Backend + Dataset |
|------|-------------------|
| `configs/milvus/triviaqa.yaml` | Milvus + TriviaQA |
| `configs/qdrant/triviaqa.yaml` | Qdrant + TriviaQA |
| `configs/pinecone/triviaqa.yaml` | Pinecone + TriviaQA |
| `configs/weaviate/triviaqa.yaml` | Weaviate + TriviaQA |
| `configs/chroma/triviaqa.yaml` | Chroma + TriviaQA |

---

**Related Documentation**:

- **Metadata Filtering** (`docs/haystack/metadata-filtering.md`): Alternative filter-focused approach
- **Multi-Tenancy** (`docs/haystack/multi-tenancy.md`): Tenant isolation (often uses namespace + filters)
- **Core Databases** (`docs/core/databases.md`): Backend wrapper details and filter formats
