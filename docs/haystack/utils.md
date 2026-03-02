# Haystack: Utils

## 1. What This Feature Is

Framework utility modules provide **shared helpers** for config, embeddings, fusion, filtering, and output shaping. This module exports seven core utility classes:

| Utility | Purpose | Use Case |
|---------|---------|----------|
| **ConfigLoader** | Config loading and validation | Pipeline initialization |
| **EmbedderFactory** | Dense/sparse embedder creation | Document/query embedding |
| **DocumentFilter** | Metadata predicate normalization | Constraint-based retrieval |
| **RerankerFactory** | Similarity/diversity ranker creation | Post-retrieval scoring |
| **ResultMerger** | RRF and weighted fusion utilities | Hybrid retrieval fusion |
| **DiversificationHelper** | Redundancy control | Diversity filtering |
| **RAGHelper** | Generator construction + prompt formatting | Answer generation |

These utilities are **framework-agnostic** and used across all Haystack feature modules.

## 2. Why It Exists in Retrieval/RAG

Without shared utilities, every feature module would need to reimplement:

| Problem | Utility Solution |
|---------|------------------|
| **Config parsing** | `ConfigLoader` with env var resolution |
| **Embedder creation** | `EmbedderFactory` with model aliases |
| **Filter normalization** | `DocumentFilter` with backend adaptation |
| **Ranker creation** | `RerankerFactory` with strategy selection |
| **Result fusion** | `ResultMerger` with RRF/weighted fusion |
| **Diversity control** | `DiversificationHelper` with MMR support |
| **RAG generation** | `RAGHelper` with prompt templates |

This module exists to:

- **Ensure consistency**: Same config loading, same embedder creation across all features
- **Reduce duplication**: One implementation, reused across 18+ feature modules
- **Centralize fixes**: Bug fixes apply to all pipelines automatically
- **Standardize interfaces**: Common types and behaviors

## 3. ConfigLoader Utility

### Purpose

Config loading with environment variable resolution and validation.

### Usage

```python
from vectordb.haystack.utils import ConfigLoader

# Load from YAML path
config = ConfigLoader.load("config.yaml")

# Load from dict
config = ConfigLoader.load_dict({"key": "value"})
```

### Environment Variable Syntax

| Syntax | Behavior |
|--------|----------|
| **`${VAR}`** | Substitute with env value, empty if unset |
| **`${VAR:-default}`** | Substitute with VAR if set, else default |

### Resolution Rules

- **Full-string only**: Only full-string placeholders resolved
- **Partial strings ignored**: `prefix_${VAR}` not interpolated
- **Recursive**: Applied to nested dicts and lists

### Example

```yaml
# config.yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "${DEVICE:-cpu}"

pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "my-index"
```

```python
import os
os.environ["DEVICE"] = "cuda"
os.environ["PINECONE_API_KEY"] = "pc-xxx"

config = ConfigLoader.load("config.yaml")
# config["embeddings"]["device"] == "cuda"
# config["pinecone"]["api_key"] == "pc-xxx"
```

## 4. EmbedderFactory Utility

### Purpose

Dense/sparse embedder creation with model aliases and warm-up.

### Usage

```python
from vectordb.haystack.utils import EmbedderFactory
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

# Create document embedder
doc_embedder = EmbedderFactory.create_document_embedder(
    model="minilm",  # Alias resolves to full HF path
    device="cpu",
    batch_size=32,
)
doc_embedder.warm_up()

# Create text embedder
text_embedder = EmbedderFactory.create_text_embedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
)
text_embedder.warm_up()

# Create sparse embedder
sparse_embedder = EmbedderFactory.create_sparse_embedder(
    model="prithvida/Splade_v2_Distilbert_uncased",
    max_length=512,
)
```

### Model Aliases

| Alias | Resolves To |
|-------|-------------|
| **qwen3** | `Qwen/Qwen3-Embedding-0.6B` |
| **minilm** | `sentence-transformers/all-MiniLM-L6-v2` |
| **mpnet** | `sentence-transformers/all-mpnet-base-v2` |

### Behavior

- **Warm-up**: Embedders must be warmed up before use
- **Device validation**: Validates device availability
- **Batch size**: Controls embedding throughput

## 5. DocumentFilter Utility

### Purpose

Metadata predicate normalization with backend adaptation and Python fallback filtering.

### Usage

```python
from vectordb.haystack.utils import DocumentFilter
from haystack import Document

# Normalize filter
normalized = DocumentFilter.normalize({"category": {"$eq": "tech"}})

# Apply filter to documents
docs = [
    Document(content="A", meta={"category": "tech"}),
    Document(content="B", meta={"category": "science"}),
]
filtered = DocumentFilter.apply_filter(docs, normalized)
```

### Supported Operators

| Operator | Description |
|----------|-------------|
| **$eq** | Equality |
| **$ne** | Not equal |
| **$gt** | Greater than |
| **$gte** | Greater than or equal |
| **$lt** | Less than |
| **$lte** | Less than or equal |
| **$in** | In list |
| **$nin** | Not in list |

### Backend Adaptation

| Backend | Filter Format |
|---------|---------------|
| **Chroma** | `{"field": {"$op": value}}` |
| **Pinecone** | MongoDB-style dict |
| **Qdrant** | `Filter(must=[...])` |
| **Milvus** | Boolean expression string |
| **Weaviate** | `Filter.by_property(...).<op>(...)` |

### Python Fallback

When backend doesn't support filter, applies Python-side filtering post-retrieval.

## 6. RerankerFactory Utility

### Purpose

Similarity/diversity ranker creation with strategy selection.

### Usage

```python
from vectordb.haystack.utils import RerankerFactory
from haystack import Document

# Create similarity ranker
ranker = RerankerFactory.create_similarity_ranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
)

# Create diversity ranker
diversity_ranker = RerankerFactory.create_diversity_ranker(
    strategy="maximum_margin_relevance",
    model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=10,
)

# Run ranker
docs = [Document(content="A", score=0.8), Document(content="B", score=0.6)]
reranked = ranker.run(query="query", documents=docs)
```

### Supported Ranker Types

| Type | Strategy | Model |
|------|----------|-------|
| **Similarity** | Cross-encoder reranking | `cross-encoder/ms-marco-*` |
| **Diversity** | Maximum Margin Relevance | `sentence-transformers/*` |
| **Diversity** | Greedy Diversity Order | `sentence-transformers/*` |

### Behavior

- **Model warm-up**: Rankers warmed up on creation
- **Top-k truncation**: Results truncated to top_k
- **Score normalization**: Scores normalized to [0,1]

## 7. ResultMerger Utility

### Purpose

RRF and weighted fusion utilities for hybrid retrieval.

### Usage (Stateless)

```python
from vectordb.haystack.utils import ResultMerger
from haystack import Document

# RRF Fusion
dense_docs = [Document(content="A", score=0.8)]
sparse_docs = [Document(content="A", score=12.0), Document(content="B", score=9.5)]

merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs, k=60, top_k=10)

# N-way RRF
ranked_lists = [
    [doc1, doc2],  # Retriever 1
    [doc2, doc3],  # Retriever 2
    [doc1, doc3],  # Retriever 3
]
merged = ResultMerger.rrf_fusion_many(ranked_lists, k=60, top_k=10)

# Weighted Fusion
merged = ResultMerger.weighted_fusion(
    dense_docs, sparse_docs,
    dense_weight=0.6, sparse_weight=0.4,
    top_k=10,
)

# Deduplication
deduped = ResultMerger.deduplicate_by_content(merged, similarity_threshold=0.95)
```

### Key Methods

| Method | Purpose | Parameters |
|--------|---------|------------|
| **stable_doc_id(doc)** | Generate consistent IDs | Precedence: `meta["doc_id"]` → `doc.id` → SHA1(content) |
| **rrf_fusion(dense, sparse, k, top_k)** | Reciprocal Rank Fusion | Default k=60 |
| **rrf_fusion_many(ranked_lists, k, top_k)** | N-way RRF | For multiple retrievers |
| **weighted_fusion(dense, sparse, weights, top_k)** | Weighted score fusion | Auto-normalizes weights |
| **deduplicate_by_content(docs, threshold)** | Exact hash dedup | Not semantic similarity |

### Behavior

- **Empty inputs**: Returns empty list
- **Duplicate handling**: Merges via stable IDs
- **Weight normalization**: Auto-normalizes if sum ≠ 1.0
- **Min-max normalization**: Normalizes scores to [0,1]

## 8. DiversificationHelper Utility

### Purpose

Redundancy control with MMR-style diversity selection.

### Usage

```python
from vectordb.haystack.utils import DiversificationHelper
from haystack import Document

helper = DiversificationHelper(
    diversity_threshold=0.7,
    max_similar_docs=3,
    mmr_lambda=0.5,
    semantic_diversification=True,
)

docs = [
    Document(content="A", score=0.9),
    Document(content="B", score=0.8),
    Document(content="C", score=0.7),
]

diverse = helper.select_diverse(docs, top_k=5)
```

### Configuration

| Parameter | Default | Impact |
|-----------|---------|--------|
| **diversity_threshold** | 0.7 | Similarity threshold for diversity |
| **max_similar_docs** | 3 | Max similar docs to keep |
| **mmr_lambda** | 0.5 | Relevance vs diversity balance |
| **semantic_diversification** | True | Use embeddings for diversity |

### Behavior

- **Threshold enforcement**: Removes docs above similarity threshold
- **MMR selection**: Balances relevance and diversity
- **Semantic diversity**: Uses embeddings when enabled

## 9. RAGHelper Utility

### Purpose

Generator construction and prompt formatting for answer generation.

### Usage

```python
from vectordb.haystack.utils import RAGHelper
from haystack import Document

# Create RAG helper
helper = RAGHelper(
    model="llama-3.3-70b-versatile",
    api_key="${GROQ_API_KEY}",
    api_base_url="https://api.groq.com/openai/v1",
    temperature=0.7,
    max_tokens=2048,
)

# Format prompt
docs = [Document(content="Context 1"), Document(content="Context 2")]
prompt = helper.format_prompt(
    query="What is RAG?",
    documents=docs,
    template="Answer based on: {context}\n\nQuestion: {query}",
)

# Generate answer
answer = helper.generate(prompt)
```

### Configuration

| Parameter | Default | Impact |
|-----------|---------|--------|
| **model** | N/A | LLM model for generation |
| **api_key** | N/A | API key (or GROQ_API_KEY env) |
| **api_base_url** | Groq default | OpenAI-compatible endpoint |
| **temperature** | 0.7 | Creativity control |
| **max_tokens** | 2048 | Output length limit |

### Behavior

- **API key fallback**: Reads GROQ_API_KEY if not provided
- **Prompt templating**: `{context}` and `{query}` placeholders
- **Error handling**: Returns None on generation failure

## 10. When to Use Utilities

### ConfigLoader

Use when:

- Loading pipeline configurations
- Need env var resolution
- Validating config structure

### EmbedderFactory

Use when:

- Creating document/query embedders
- Need model alias resolution
- Consistent embedder initialization

### DocumentFilter

Use when:

- Normalizing filter predicates
- Backend filter adaptation needed
- Python fallback filtering required

### RerankerFactory

Use when:

- Creating similarity/diversity rankers
- Strategy selection needed
- Consistent ranker initialization

### ResultMerger

Use when:

- Fusing dense+sparse results
- Multiple retriever outputs need merging
- Deduplication across sources

### DiversificationHelper

Use when:

- Controlling result redundancy
- MMR-style diversity needed
- Semantic diversity required

### RAGHelper

Use when:

- Generating answers from context
- Prompt formatting needed
- OpenAI-compatible generation

## 11. When Not to Use Utilities

### Avoid When

| Utility | Avoid When |
|---------|------------|
| **ConfigLoader** | Custom config format needed |
| **EmbedderFactory** | Custom embedder logic required |
| **DocumentFilter** | Backend-native filters sufficient |
| **RerankerFactory** | Custom ranking logic needed |
| **ResultMerger** | Single retriever, no fusion needed |
| **DiversificationHelper** | Diversity not required |
| **RAGHelper** | Custom generation pipeline |

## 12. Failure Modes and Edge Cases

### ConfigLoader

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing config file** | `FileNotFoundError` | Verify path |
| **Malformed YAML** | Parse error | Validate YAML |
| **Missing env var** | Empty string or default | Use `${VAR:-default}` |

### EmbedderFactory

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing API key** | Raises `ValueError` | Set env var |
| **Invalid model** | Model load failure | Verify model name |
| **Device unavailable** | Falls back to CPU | Check device |

### DocumentFilter

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Invalid operator** | Python fallback | Use supported operators |
| **Backend filter error** | Python fallback | Accept fallback |

### RerankerFactory

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Model load failure** | Raises error | Verify model |
| **OOM error** | Use smaller model | Select lighter model |

### ResultMerger

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Empty inputs** | Returns empty list | Not an error |
| **Duplicate content** | Merges via stable IDs | Expected behavior |

### DiversificationHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **No diverse docs** | Returns available | Accept partial |
| **Embedding failure** | Falls back to non-semantic | Check embeddings |

### RAGHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing API key** | Raises `ValueError` | Set GROQ_API_KEY |
| **Generation error** | Returns None | Check API status |

## 13. Practical Usage Examples

### Example 1: Complete Utility Pipeline

```python
from vectordb.haystack.utils import (
    ConfigLoader,
    EmbedderFactory,
    DocumentFilter,
    RerankerFactory,
    ResultMerger,
    RAGHelper,
)

# Load config
config = ConfigLoader.load("config.yaml")

# Create embedders
doc_embedder = EmbedderFactory.create_document_embedder(
    model=config["embeddings"]["model"],
    device=config["embeddings"]["device"],
)
doc_embedder.warm_up()

# Create filter
filter_spec = DocumentFilter.normalize({"category": {"$eq": "tech"}})

# Create ranker
ranker = RerankerFactory.create_similarity_ranker(
    model=config["reranker"]["model"],
    top_k=config["reranker"]["top_k"],
)

# Create RAG helper
rag_helper = RAGHelper(
    model=config["generator"]["model"],
    api_key=config["generator"]["api_key"],
)

# Use utilities together
docs = [...]  # Retrieved documents
filtered = DocumentFilter.apply_filter(docs, filter_spec)
reranked = ranker.run(query="query", documents=filtered)
merged = ResultMerger.rrf_fusion(reranked, [], top_k=10)
answer = rag_helper.generate(rag_helper.format_prompt("query", merged))
```

### Example 2: Hybrid Retrieval Fusion

```python
from vectordb.haystack.utils import ResultMerger

# Dense and sparse results
dense_results = [
    {"id": "1", "score": 0.8, "content": "A"},
    {"id": "2", "score": 0.6, "content": "B"},
]
sparse_results = [
    {"id": "1", "score": 12.0, "content": "A"},
    {"id": "3", "score": 9.5, "content": "C"},
]

# Convert to Documents
from haystack import Document
dense_docs = [Document(content=d["content"], score=d["score"], id=d["id"]) for d in dense_results]
sparse_docs = [Document(content=d["content"], score=d["score"], id=d["id"]) for d in sparse_results]

# Fuse with RRF
merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs, k=60, top_k=10)
```

### Example 3: Config with Env Vars

```yaml
# config.yaml
embeddings:
  model: "${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
  device: "${DEVICE:-cpu}"

generator:
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  api_base_url: "${API_BASE_URL:-https://api.groq.com/openai/v1}"
```

```python
import os
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-mpnet-base-v2"
os.environ["DEVICE"] = "cuda"

config = ConfigLoader.load("config.yaml")
# config["embeddings"]["model"] == "sentence-transformers/all-mpnet-base-v2"
# config["embeddings"]["device"] == "cuda"
```

## 14. Source Walkthrough Map

### Primary Module Files

| File | Purpose |
|------|---------|
| `src/vectordb/haystack/utils/__init__.py` | Public API exports |
| `src/vectordb/haystack/utils/README.md` | Feature overview |

### Utility Implementations

| File | Utility |
|------|---------|
| `config.py` | ConfigLoader (re-exports from vectordb.utils) |
| `embeddings.py` | EmbedderFactory |
| `filters.py` | DocumentFilter |
| `reranker.py` | RerankerFactory |
| `fusion.py` | ResultMerger |
| `diversification.py` | DiversificationHelper |
| `rag.py` | RAGHelper |

### Test Files

| File | Coverage |
|------|----------|
| `tests/haystack/utils/test_embeddings.py` | EmbedderFactory tests |
| `tests/haystack/utils/test_reranker.py` | RerankerFactory tests |
| `tests/haystack/utils/test_fusion.py` | ResultMerger tests |
| `tests/haystack/utils/test_filters.py` | DocumentFilter tests |
| `tests/haystack/utils/test_diversification.py` | DiversificationHelper tests |
| `tests/haystack/utils/test_config.py` | ConfigLoader tests |
| `tests/haystack/utils/test_rag.py` | RAGHelper tests |

---

**Related Documentation**:

- **Components** (`docs/haystack/components.md`): Reusable advanced-RAG components
- **Core Shared Utils** (`docs/core/shared-utils.md`): Cross-framework utilities
- **Reference Config** (`docs/reference/config-reference.md`): Configuration key inventory
