# Haystack: Metadata Filtering

## 1. What This Feature Is

Metadata filtering combines semantic vector search with structured metadata constraints to narrow retrieval results. This module provides two implementation layers:

### Layer 1: Modular Production Pipelines

Located in `indexing/` and `search/` subdirectories, these use:

- Unified VectorDB wrappers (`vectordb.databases.*`)
- Shared utilities in `common/` (config, filters, embeddings, RAG)
- Consistent pipeline interface across all backends

### Layer 2: Native Backend-Specific Pipelines

Located in top-level backend files (`chroma.py`, `milvus.py`, etc.), these provide:

- Explicit backend-specific filter expression builders
- Pre-filter timing and candidate counting
- `SelectivityAnalyzer` for offline selectivity estimation

Both layers model filters as structured conditions over document metadata (category, year, ticker, title) and apply constraints during retrieval.

## 2. Why It Exists in Retrieval/RAG

Pure vector similarity is often too broad for production use cases. Metadata filtering addresses three critical needs:

| Need | Solution |
|------|----------|
| **Precision** | Narrow candidate set to scoped tasks (tenant, domain, time window, source type) |
| **Efficiency** | Reduce search space when corpus is large and constraints are selective |
| **Controllability** | Enable reproducible evaluation with explicit deterministic metadata slices |

This mirrors official guidance across Haystack and vector databases: **filters are a first-class retrieval input, not a post-hoc cleanup step**.

### Common Use Cases

- **Domain slicing**: `category == "science"`
- **Temporal filtering**: `year >= 2022`
- **Source control**: `source in ["wikipedia", "arxiv"]`
- **Title matching**: `title contains "transformer"`
- **Finance fields**: `ticker == "AAPL"`, `quarter == "Q4"`, `speaker == "CEO"`

## 3. Indexing Pipeline: Step-by-Step

```mermaid
flowchart TD
    A[Config Loading] --> B[Validate Required Sections]
    B --> C[Initialize Backend Wrapper]
    C --> D[Load Documents from Dataloader]
    D --> E{Documents Empty?}
    E -->|Yes| F[Raise ValueError]
    E -->|No| G[Embed Documents]
    G --> H[Create Collection/Index]
    H --> I[Write Documents with Metadata]
    I --> J[Return documents_indexed Count]
```

### Modular Indexing Flow (`*MetadataFilteringIndexingPipeline`)

1. **Load config with env resolution**:
   - `load_metadata_filtering_config()` accepts dict or YAML path
   - `${VAR}` and `${VAR:-default}` resolved recursively

2. **Validate required sections**:
   - Each backend enforces `dataloader`, `embeddings`, and backend section

3. **Initialize backend wrapper**:
   - `_init_db()` creates `vectordb.databases.*VectorDB` instance

4. **Load documents**:
   - `load_documents_from_config()` uses `DataloaderCatalog.create(...)`
   - Returns Haystack `Document` objects
   - Empty loads raise `ValueError`

5. **Embed documents**:
   - `get_document_embedder()` builds and warms `SentenceTransformersDocumentEmbedder`
   - `embedder.run(documents=...)` returns embedded docs

6. **Create collection/index**:
   - Chroma/Weaviate/Milvus/Qdrant: `create_collection(...)`
   - Pinecone: `create_index(...)`

7. **Write documents**:
   - Chroma: `add_documents(...)`
   - Milvus: `insert_documents(...)`
   - Qdrant: `index_documents(...)`
   - Weaviate: `insert_documents(...)`
   - Pinecone: `upsert(...)`

8. **Return summary**: `{"documents_indexed": <count>}`

## 4. Search Pipeline: Step-by-Step

```mermaid
flowchart TD
    A[Config Load + Validate] --> B[Resolve Query + top_k]
    B --> C[Embed Query Text]
    C --> D[Parse Filter from Config]
    D --> E[Convert to Canonical Dict]
    E --> F[Execute Backend Search with Filter]
    F --> G[Build FilteredQueryResult List]
    G --> H{RAG Enabled?}
    H -->|Yes| I[Generate Answer]
    H -->|No| J[Return Retrieval Results]
    I --> J
```

### Modular Search Flow (`*MetadataFilteringSearchPipeline`)

1. **Load and validate config**:
   - Requires `embeddings`, backend section, and `search`

2. **Resolve query and top_k**:
   - Query argument overrides config
   - Fallback: `metadata_filtering.test_query` or `"test query"`
   - `top_k` from `search.top_k` (default 10)

3. **Embed query text**:
   - `get_text_embedder()` builds and warms `SentenceTransformersTextEmbedder`
   - `run(text=query)` returns query embedding

4. **Parse filter spec**:
   - `common.filters.parse_filter_from_config()` reads first `metadata_filtering.test_filters` item
   - Returns empty `FilterSpec` if missing (no exception)

5. **Convert to canonical dict**:
   - `filter_spec_to_canonical_dict()` returns:
     - Single condition: `{"field": "...", "operator": "...", "value": "..."}`
     - Multi-condition: `{"operator": "and", "conditions": [...]}`

6. **Execute backend search**:
   - Chroma: `db.query(query_embedding=..., top_k=..., where=...)`
   - Qdrant/Milvus/Weaviate: `db.search(query_embedding=..., top_k=..., filters=...)`
   - Pinecone: `db.query(vector=..., top_k=..., filter=...)`

7. **Build results**:
   - Relevance: `doc.score` (fallback `0.0` if `None`)
   - Rank starts at 1
   - Timing attached only to rank 1 result
   - `pre_filter_ms` is `0.0` placeholder in modular search

8. **Optional RAG**:
   - `create_rag_generator()` creates `OpenAIGenerator` when `rag.enabled=true`
   - `generate_answer()` uses top documents for context

## 5. When to Use It

Use metadata filtering when:

- **Retrieval quality depends on known constraints**: Query-time metadata filters improve precision
- **Domain-specific slicing needed**: Category, time window, source type filtering
- **Large corpus with selective constraints**: Filtering before/during search improves efficiency
- **Reproducible benchmarks**: Fixed `test_query` + `test_filters` for eval runs
- **Multi-tenant scenarios**: Combine with tenant ID filtering for isolation
- **Structured data requirements**: Finance, legal, scientific domains with typed metadata

## 6. When Not to Use It

Avoid metadata filtering when:

- **Metadata is sparse or noisy**: Inconsistent metadata leads to unpredictable results
- **Fuzzy intent extraction needed**: Hard-coded filters don't match vague user intent
- **Complex boolean logic required**: Some backends limit filter expression complexity
- **Two filter stacks confusion**: This codebase has canonical dict vs native expression builders; mixing causes issues
- **Post-filtering acceptable**: For small corpora, filtering after retrieval may be simpler

## 7. What This Codebase Provides

### Core Shared Modules (`common/`)

```python
from vectordb.haystack.metadata_filtering.common.config import (
    load_metadata_filtering_config,  # Config loading with env resolution
)
from vectordb.haystack.metadata_filtering.common.dataloader import (
    load_documents_from_config,  # Dataset → Haystack Documents
)
from vectordb.haystack.metadata_filtering.common.embeddings import (
    get_document_embedder,  # Document embedder creation
    get_text_embedder,     # Query embedder creation
)
from vectordb.haystack.metadata_filtering.common.filters import (
    parse_filter_from_config,     # Config → FilterSpec
    filter_spec_to_canonical_dict, # FilterSpec → canonical dict
)
from vectordb.haystack.metadata_filtering.common.rag import (
    create_rag_generator,  # Optional RAG setup
    generate_answer,       # Answer generation
)
from vectordb.haystack.metadata_filtering.common.types import (
    FilterSpec,           # Filter specification
    FilterCondition,      # Single filter condition
    FilteredQueryResult,  # Search result with timing
)
from vectordb.haystack.metadata_filtering.common.timer import (
    Timer,  # Timing context manager
)
```

### Modular Pipeline Classes

**Indexing**:

```python
from vectordb.haystack.metadata_filtering.indexing import (
    MilvusMetadataFilteringIndexingPipeline,
    QdrantMetadataFilteringIndexingPipeline,
    PineconeMetadataFilteringIndexingPipeline,
    ChromaMetadataFilteringIndexingPipeline,
    WeaviateMetadataFilteringIndexingPipeline,
)
```

**Search**:

```python
from vectordb.haystack.metadata_filtering.search import (
    MilvusMetadataFilteringSearchPipeline,
    QdrantMetadataFilteringSearchPipeline,
    PineconeMetadataFilteringSearchPipeline,
    ChromaMetadataFilteringSearchPipeline,
    WeaviateMetadataFilteringSearchPipeline,
)
```

### Native Expression Stack (`vectordb_pipeline_type.py`)

```python
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    # Types
    "FilterField",
    "FilterCondition",
    "FilterSpec",
    "TimingMetrics",
    "FilteredQueryResult",

    # Backend-specific builders
    "MilvusFilterExpressionBuilder",
    "QdrantFilterExpressionBuilder",
    "PineconeFilterExpressionBuilder",
    "WeaviateFilterExpressionBuilder",
    "ChromaFilterExpressionBuilder",

    # Validation and parsing
    "parse_filter_from_config",
    "validate_filter_config",

    # Selectivity analysis
    "SelectivityAnalyzer",
)
```

### Top-Level Backend Pipelines

```python
from vectordb.haystack.metadata_filtering import (
    MilvusMetadataFilteringPipeline,
    QdrantMetadataFilteringPipeline,
    PineconeMetadataFilteringPipeline,
    ChromaMetadataFilteringPipeline,
    WeaviateMetadataFilteringPipeline,
)
```

## 8. Backend-Specific Behavior Differences

### Filter Expression Formats

| Backend | Format | Example |
|---------|--------|---------|
| **Milvus** | Boolean expression string | `'metadata["category"] == "tech" && metadata["year"] > 2020'` |
| **Qdrant** | `models.Filter(must=[...])` | `Filter(must=[FieldCondition(key="category", match=MatchValue(value="tech"))])` |
| **Pinecone** | MongoDB-style dict | `{"$and": [{"category": {"$eq": "tech"}}, {"year": {"$gt": 2020}}]}` |
| **Weaviate** | Where-clause dict | `{"operator": "And", "operands": [{"path": ["category"], "operator": "Equal", "valueText": "tech"}]}` |
| **Chroma** | Mongo-style dict | `{"$and": [{"category": {"$eq": "tech"}}]}` |

### Modular vs Native Layer Differences

| Aspect | Modular Layer | Native Layer |
|--------|---------------|--------------|
| **Filter input** | Canonical dict from `common.filters` | Native expression builders |
| **Empty filters** | Passed as `None` to backend | Validated strictly, raises if missing |
| **Candidate counts** | `-1` placeholders | Actual counts reported |
| **Pre-filter timing** | `0.0` placeholder | Measured and reported |
| **Exception handling** | Propagates backend exceptions | Catches and returns `0` for pre-filter |

### Operational Differences

| Backend | Notes |
|---------|-------|
| **Pinecone** | Explicit `namespace` handling in indexing/search |
| **Weaviate** | PascalCase collection names common in configs |
| **Chroma** | `where=None` passed for empty filters in modular search |
| **Qdrant** | `Filter` objects with `FieldCondition` for complex queries |
| **Milvus** | JSON path notation `metadata["field"]` in expressions |

## 9. Configuration Semantics

### Primary Configuration Keys

```yaml
# Dataloader configuration
dataloader:
  type: "huggingface"
  dataset_name: "trivia_qa"
  config: "rc"
  split: "test"
  limit: 500

# Embedding configuration (must match between indexing and search)
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  trust_remote_code: false

# Backend configuration (example: Qdrant)
qdrant:
  url: "http://localhost:6333"
  api_key: null
  collection_name: "metadata-filtering-demo"

# Search configuration
search:
  top_k: 10

# Metadata filtering configuration
metadata_filtering:
  schema:
    allowed_fields: ["category", "year", "source", "title"]
    allowed_operators: ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in"]

  test_query: "What is the capital of France?"
  test_filters:
    - field: "category"
      operator: "$eq"
      value: "science"
    - field: "year"
      operator: "$gte"
      value: 2020

# RAG configuration (optional)
rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  api_base_url: "https://api.groq.com/openai/v1"
  temperature: 0.7
  max_tokens: 2048

# Logging configuration
logging:
  level: "INFO"
  name: "metadata-filtering-pipeline"
```

### Filter Schema Definition

```yaml
metadata_filtering:
  schema:
    allowed_fields:
      - "category"
      - "year"
      - "quarter"
      - "ticker"
      - "speaker"
      - "source"
      - "title"
    allowed_operators:
      - "$eq"   # Equality
      - "$ne"   # Not equal
      - "$gt"   # Greater than
      - "$gte"  # Greater than or equal
      - "$lt"   # Less than
      - "$lte"  # Less than or equal
      - "$in"   # In list
      - "$nin"  # Not in list
```

### Environment Variable Syntax

- `${VAR}`: Substitute with env var, empty string if unset
- `${VAR:-default}`: Substitute with VAR if set, else default

```yaml
qdrant:
  url: "${QDRANT_URL:-http://localhost:6333}"
  api_key: "${QDRANT_API_KEY}"  # Required, no default
```

## 10. Failure Modes and Edge Cases

### Configuration Failures

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Missing required sections** | `dataloader`, `embeddings`, or backend section missing | Raises `ValueError` in constructor |
| **Invalid operator** | Operator not in allowed list | Raises `ValueError` in filter validation |
| **Empty YAML file** | `load_config()` returns `None` | Downstream code expects dict; add content or validate |

### Indexing Failures

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Empty document load** | Dataset empty or limit=0 | Raises `ValueError("No documents loaded...")` |
| **Embedding dimension mismatch** | Index dimension ≠ embedding dimension | Verify `embeddings.dimension` matches index config |
| **Backend connection error** | Invalid URL, API key, network issue | Validate credentials; check connectivity |

### Search Failures

| Failure | Cause | Mitigation |
|---------|-------|------------|
| **Missing test_filters** | Native parser requires filters | Modular parser returns empty spec; native raises |
| **Empty filter behavior** | Backend-dependent handling | Chroma passes `where=None`; others vary |
| **doc.score is None** | Backend doesn't return scores | Normalized to `0.0` in result mapping |
| **RAG without model** | `rag.enabled=true` but no `rag.model` | Raises `ValueError`; generation returns `None` answer |

### Timing and Result Edge Cases

| Issue | Cause | Mitigation |
|-------|-------|------------|
| **Timing only on rank 1** | By design; reduces overhead | Accept limitation; extend if needed |
| **Candidate counts = -1** | Modular search doesn't track | Use native layer for accurate counts |
| **pre_filter_ms = 0.0** | Modular search doesn't measure | Use native layer for timing breakdown |

### Native Pipeline Limitations

| Issue | Cause | Mitigation |
|-------|-------|------------|
| **run() skips indexing** | Initializes empty `documents` list | Extend code to load documents properly |
| **_pre_filter() catches exceptions** | Returns `0` on error | Check logs for underlying issues |

## 11. Practical Usage Examples

### Example 1: Indexing with Qdrant

```python
from vectordb.haystack.metadata_filtering.indexing import (
    QdrantMetadataFilteringIndexingPipeline,
)

# Initialize pipeline
pipeline = QdrantMetadataFilteringIndexingPipeline(
    "src/vectordb/haystack/metadata_filtering/configs/qdrant_arc.yaml"
)

# Run indexing
summary = pipeline.run()
print(f"Indexed {summary['documents_indexed']} documents")
```

### Example 2: Search with Pinecone

```python
from vectordb.haystack.metadata_filtering.search import (
    PineconeMetadataFilteringSearchPipeline,
)

# Initialize pipeline
pipeline = PineconeMetadataFilteringSearchPipeline(
    "src/vectordb/haystack/metadata_filtering/configs/pinecone_triviaqa.yaml"
)

# Search with filter from config
results = pipeline.search("What is the capital of France?")

for result in results:
    print(f"Rank {result.rank}: Score {result.relevance} - {result.doc.content[:100]}")
```

### Example 3: Search with Custom Query

```python
from vectordb.haystack.metadata_filtering.search import (
    WeaviateMetadataFilteringSearchPipeline,
)

pipeline = WeaviateMetadataFilteringSearchPipeline(
    "src/vectordb/haystack/metadata_filtering/configs/weaviate_earnings_calls.yaml"
)

# Override query and top_k
results = pipeline.search(
    query="What was the revenue growth?",
    top_k=5,
)
```

### Example 4: Search with RAG Enabled

```yaml
# In config.yaml
rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  prompt_template: |
    Based on the following context, answer the question.

    Context:
    {context}

    Question: {query}

    Answer:
```

```python
from vectordb.haystack.metadata_filtering.search import (
    ChromaMetadataFilteringSearchPipeline,
)

pipeline = ChromaMetadataFilteringSearchPipeline("config.yaml")

# Search returns results; RAG generates answer separately
results = pipeline.search("What is quantum computing?")
answer = pipeline.generate_answer(results)  # Uses RAG pipeline
print(f"Answer: {answer}")
```

### Example 5: Native Layer with Selectivity Analysis

```python
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    SelectivityAnalyzer,
    parse_filter_from_config,
)

# Parse filter from config
config = {
    "metadata_filtering": {
        "test_filters": [
            {"field": "category", "operator": "$eq", "value": "science"},
            {"field": "year", "operator": "$gte", "value": 2020},
        ]
    }
}
filter_spec = parse_filter_from_config(config)

# Analyze selectivity (offline estimation)
analyzer = SelectivityAnalyzer(total_docs=10000)
selectivity = analyzer.estimate_selectivity(filter_spec)
print(f"Estimated selectivity: {selectivity:.2%}")
# Higher selectivity = more selective filter = fewer candidates
```

### Example 6: Multi-Condition Filters

```yaml
# Config with complex filter
metadata_filtering:
  test_filters:
    - operator: "and"
      conditions:
        - field: "category"
          operator: "$eq"
          value: "technology"
        - operator: "or"
          conditions:
            - field: "year"
              operator: "$gte"
              value: 2022
            - field: "source"
              operator: "$in"
              value: ["arxiv", "wikipedia"]
```

```python
from vectordb.haystack.metadata_filtering.search import (
    MilvusMetadataFilteringSearchPipeline,
)

pipeline = MilvusMetadataFilteringSearchPipeline("config.yaml")
results = pipeline.search()  # Uses complex filter from config
```

## 12. Source Walkthrough Map

### Primary Entrypoints

| File | Purpose |
|------|---------|
| `src/vectordb/haystack/metadata_filtering/__init__.py` | Public API exports |
| `src/vectordb/haystack/metadata_filtering/indexing/__init__.py` | Indexing pipeline exports |
| `src/vectordb/haystack/metadata_filtering/search/__init__.py` | Search pipeline exports |

### Shared Utilities (`common/`)

| File | Key Components |
|------|----------------|
| `common/config.py` | `load_metadata_filtering_config`, env resolution |
| `common/dataloader.py` | `load_documents_from_config` |
| `common/embeddings.py` | `get_document_embedder`, `get_text_embedder` |
| `common/filters.py` | `parse_filter_from_config`, `filter_spec_to_canonical_dict` |
| `common/rag.py` | `create_rag_generator`, `generate_answer` |
| `common/types.py` | `FilterSpec`, `FilterCondition`, `FilteredQueryResult` |
| `common/timer.py` | `Timer` context manager |

### Modular Indexing Pipelines

| File | Backend |
|------|---------|
| `indexing/chroma.py` | Chroma |
| `indexing/milvus.py` | Milvus |
| `indexing/pinecone.py` | Pinecone |
| `indexing/qdrant.py` | Qdrant |
| `indexing/weaviate.py` | Weaviate |

### Modular Search Pipelines

| File | Backend |
|------|---------|
| `search/chroma.py` | Chroma |
| `search/milvus.py` | Milvus |
| `search/pinecone.py` | Pinecone |
| `search/qdrant.py` | Qdrant |
| `search/weaviate.py` | Weaviate |

### Native Expression Stack

| File | Purpose |
|------|---------|
| `base.py` | Base class for native pipelines |
| `vectordb_pipeline_type.py` | Filter builders, types, selectivity analyzer |
| `chroma.py` | Native Chroma pipeline |
| `milvus.py` | Native Milvus pipeline |
| `pinecone.py` | Native Pinecone pipeline |
| `qdrant.py` | Native Qdrant pipeline |
| `weaviate.py` | Native Weaviate pipeline |

### Configuration Examples

| File | Backend + Dataset |
|------|-------------------|
| `configs/chroma_arc.yaml` | Chroma + ARC |
| `configs/milvus_triviaqa.yaml` | Milvus + TriviaQA |
| `configs/pinecone_triviaqa.yaml` | Pinecone + TriviaQA |
| `configs/qdrant_arc.yaml` | Qdrant + ARC |
| `configs/weaviate_earnings_calls.yaml` | Weaviate + Earnings Calls |

### Test Files

| Directory | Coverage |
|-----------|----------|
| `tests/haystack/metadata_filtering/test_common/` | Shared utilities tests |
| `tests/haystack/metadata_filtering/test_indexing/` | Indexing pipeline tests |
| `tests/haystack/metadata_filtering/test_search/` | Search pipeline tests |
| `tests/haystack/metadata_filtering/test_vectordb_pipeline_type.py` | Filter builder tests |

---

**Related Documentation**:

- **Multi-Tenancy** (`docs/haystack/multi-tenancy.md`): Tenant isolation (often combined with metadata filtering)
- **Hybrid Indexing** (`docs/haystack/hybrid-indexing.md`): Dense+sparse retrieval (alternative to filtering)
- **Core Databases** (`docs/core/databases.md`): Backend wrapper filter formats
