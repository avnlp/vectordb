# Utils

Shared utilities used by backend wrappers and feature pipelines for consistent configuration loading, evaluation, sparse embedding handling, document ID management, multi-tenancy scope injection, output structuring, and logging.

## What This Module Covers

### Configuration (`config.py`, `config_loader.py`)

Two complementary configuration utilities:

**`config.py`** provides standalone functions used by the database wrappers:

- `load_config(path)`: Loads a YAML file and resolves environment variables. Supports `${VAR}` and `${VAR:-default}` syntax.
- `resolve_env_vars(value)`: Recursively resolves environment variable patterns in strings, dicts, and lists.
- `setup_logger(config)`: Creates a logger from the `logging` section of a config dict.
- `resolve_embedding_model(name)`: Resolves short model aliases to full HuggingFace model paths. Known aliases: `"qwen3"` → `"Qwen/Qwen3-Embedding-0.6B"`, `"minilm"` → `"sentence-transformers/all-MiniLM-L6-v2"`, `"mpnet"` → `"sentence-transformers/all-mpnet-base-v2"`.
- `get_dataset_limits(name)`: Returns default `index_limit` and `eval_limit` for a dataset name.

**`config_loader.py`** provides `ConfigLoader`, a class used by Haystack and LangChain feature pipelines:

- `ConfigLoader.load(config_or_path)`: Accepts a dict or a YAML file path, resolves environment variables, and returns the config dict.
- `ConfigLoader.validate(config, db_type)`: Checks that the config contains the required sections (`dataloader`, `embeddings`, and the specified database type key).

```python
from vectordb.utils.config_loader import ConfigLoader

config = ConfigLoader.load("configs/pinecone_triviaqa.yaml")
ConfigLoader.validate(config, "pinecone")
```

---

### Evaluation Metrics (`evaluation.py`)

Standard information retrieval metrics for measuring pipeline quality against ground-truth relevant document IDs.

**Metrics:**

| Metric | Formula | What It Measures |
|---|---|---|
| `Recall@k` | \|relevant ∩ top_k\| / \|relevant\| | Fraction of relevant docs retrieved |
| `Precision@k` | \|relevant ∩ top_k\| / k | Fraction of top-k that are relevant |
| `MRR` | 1 / rank_of_first_relevant | How early the first relevant doc appears |
| `NDCG@k` | DCG@k / IDCG@k | Rank-aware relevance normalized to ideal |
| `Hit Rate` | 1 if any relevant in top_k, else 0 | Binary success indicator |

**Usage:**

```python
from vectordb.utils.evaluation import evaluate_retrieval, QueryResult

query_results = [
    QueryResult(
        query="What is machine learning?",
        retrieved_ids=["doc1", "doc2", "doc3"],
        relevant_ids={"doc1", "doc5"},
    ),
]
metrics = evaluate_retrieval(query_results, k=5)
print(metrics.to_dict())
# {"recall@5": 0.5, "precision@5": 0.33, "mrr": 1.0, "ndcg@5": ..., "hit_rate": 1.0}
```

**Output containers:**

- `RetrievalMetrics`: Aggregated metric values across all queries, with `to_dict()` for serialization.
- `QueryResult`: Per-query results with `retrieved_ids`, `relevant_ids`, and `scores`.
- `EvaluationResult`: Full evaluation run with metrics, per-query results, pipeline name, and dataset name.

---

### Sparse Embedding Utilities (`sparse.py`)

Bidirectional conversion between the Haystack `SparseEmbedding` standard format and backend-native formats.

| Function | Purpose |
|---|---|
| `normalize_sparse(sparse)` | Convert any format to `SparseEmbedding`. Accepts `SparseEmbedding`, Milvus `{int: float}`, Pinecone `{"indices": [...], "values": [...]}`, or `None`. |
| `to_milvus_sparse(sparse)` | Convert `SparseEmbedding` to Milvus `{index: value}` format. |
| `to_pinecone_sparse(sparse)` | Convert `SparseEmbedding` to Pinecone `{"indices": [...], "values": [...]}` format. |
| `to_qdrant_sparse(sparse)` | Convert `SparseEmbedding` to Qdrant `SparseVector` object. |
| `get_doc_sparse_embedding(doc)` | Extract sparse embedding from a Haystack Document, checking both `doc.sparse_embedding` and `doc.meta["sparse_embedding"]` for backward compatibility. |

```python
from vectordb.utils.sparse import normalize_sparse, to_pinecone_sparse

# From Milvus format to Haystack
sparse = normalize_sparse({1: 0.5, 5: 0.8, 23: 0.3})
# sparse = SparseEmbedding(indices=[1, 5, 23], values=[0.5, 0.8, 0.3])

# To Pinecone format
pinecone_fmt = to_pinecone_sparse(sparse)
# pinecone_fmt = {"indices": [1, 5, 23], "values": [0.5, 0.8, 0.3]}
```

---

### Document ID Utilities (`ids.py`)

Centralized document ID management ensuring consistent ID handling across all vector database integrations.

| Function | Purpose |
|---|---|
| `get_doc_id(doc)` | Extract string ID from a Haystack Document. Priority: `doc.id` → `doc.meta["doc_id"]` → auto-generated UUID4. |
| `set_doc_id(doc, doc_id)` | Set ID on both `doc.id` and `doc.meta["doc_id"]` to survive serialization round-trips. |
| `coerce_id(value)` | Convert any value (int, UUID, string, None) to a string ID. |

IDs are set in two locations (`doc.id` and `doc.meta["doc_id"]`) to ensure they survive storage backends that may only preserve metadata fields. All backend wrappers use `get_doc_id()` for upsert operations.

---

### Scope and Namespace Injection (`scope.py`)

Utilities for implementing data isolation in multi-tenant and namespace deployments. These functions are called by feature pipelines before indexing and querying to inject tenant context into metadata and filters.

| Function | Purpose |
|---|---|
| `inject_scope_to_metadata(metadata, scope)` | Add `tenant_id` (or custom field) to a metadata dictionary for use during document indexing. |
| `inject_scope_to_filter(filters, scope)` | Add a scope equality condition to an existing filter dict using `$and` + `$eq` operators. |
| `build_scope_filter_expr(scope, field, existing_expr)` | Build a Milvus-style boolean expression string that combines an existing expression with a scope equality condition. |

```python
from vectordb.utils.scope import inject_scope_to_metadata, inject_scope_to_filter

# During indexing
doc.meta = inject_scope_to_metadata(doc.meta, "tenant_a")
# Result: {"category": "news", "tenant_id": "tenant_a"}

# During querying
filters = inject_scope_to_filter({"category": "news"}, "tenant_a")
# Result: {"$and": [{"tenant_id": {"$eq": "tenant_a"}}, {"category": "news"}]}
```

---

### Output Structures (`output.py`)

Standardized dataclasses for retrieval results that provide consistent output format across all pipeline implementations.

- **`RetrievedDocument`**: Individual document with `content`, `doc_id`, `score`, `metadata`, and optional `matched_children` (for parent-child retrieval).
- **`RetrievalOutput`**: Complete result from one query with query text, documents, retrieval mode, top_k, total retrieved, and latency in milliseconds.
- **`PipelineOutput`**: Full pipeline run output with pipeline name, database type, dataset name, indexing stats, retrieval results, and evaluation metrics. The `summary()` method returns a human-readable formatted string.

All containers implement `to_dict()` for JSON serialization.

---

### Document Converters (`*_document_converter.py`)

Per-backend converters that transform Haystack and LangChain documents to and from each database's native storage format.

| File | Converts For |
|---|---|
| `chroma_document_converter.py` | Chroma batch format (`{"texts": [...], "embeddings": [...], "metadatas": [...], "ids": [...]}`) |
| `pinecone_document_converter.py` | Pinecone vector records with `id`, `values`, `sparse_values`, and `metadata` |
| `qdrant_document_converter.py` | Qdrant `PointStruct` objects with payloads and named vectors |
| `weaviate_document_converter.py` | Weaviate object dictionaries with `vector` and property fields |

Chroma stores list-of-string metadata values as `"$ $"`-delimited strings because Chroma's metadata indexing requires scalar values.

---

### Logging (`logging.py`)

`LoggerFactory` provides singleton-style logger initialization to prevent duplicate handlers across modules. Each module creates its own logger via:

```python
from vectordb.utils.logging import LoggerFactory
logger = LoggerFactory(__name__, log_level=logging.INFO).get_logger()
```

The `configure_from_env()` class method reads the log level from the `LOG_LEVEL` environment variable, defaulting to `INFO`.

---

## Common Pitfalls

- Duplicating utility logic (for example, embedding conversion or scope injection) inside feature modules instead of calling the shared helpers. This causes inconsistent behavior when the shared implementation changes.
- Changing default values in `config.py` (such as `DEFAULT_EMBEDDING_MODEL` or `DATASET_LIMITS`) without verifying that existing configs and tests still produce the same results.
- Using framework-specific assumptions (Haystack vs LangChain document fields) inside utilities that are intended to be framework-agnostic. The `ids.py` and `scope.py` modules intentionally use duck typing to work with both frameworks.
