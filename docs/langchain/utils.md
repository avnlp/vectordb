# LangChain: Utils

## 1. What This Feature Is

Framework utility modules provide **shared helpers** for config, embeddings, fusion, filtering, and output shaping. This module exports core utility classes:

| Utility | Purpose | Use Case |
|---------|---------|----------|
| **ConfigLoader** | Config loading and validation | Pipeline initialization |
| **EmbedderHelper** | Dense embedder creation | Document/query embedding |
| **SparseEmbedder** | Sparse embedder creation | Hybrid/sparse retrieval |
| **RerankerHelper** | Cross-encoder reranker creation | Post-retrieval scoring |
| **MMRHelper** | MMR diversity algorithm | Diversity filtering |
| **ResultMerger** | RRF and weighted fusion utilities | Hybrid retrieval fusion |
| **RAGHelper** | Generator construction + prompt formatting | Answer generation |
| **DocumentConverter** | LangChain ↔ backend format conversion | Document format conversion |
| **FiltersHelper** | MongoDB-style filter translation | Constraint-based retrieval |
| **DiversificationHelper** | Redundancy control | Diversity filtering |

These utilities are **framework-agnostic** and used across all LangChain feature modules.

## 2. Why It Exists in Retrieval/RAG

Without shared utilities, every feature module would need to reimplement:

| Problem | Utility Solution |
|---------|------------------|
| **Config parsing** | `ConfigLoader` with env var resolution |
| **Embedder creation** | `EmbedderHelper` with model aliases |
| **Filter normalization** | `FiltersHelper` with backend adaptation |
| **Ranker creation** | `RerankerHelper` with strategy selection |
| **Result fusion** | `ResultMerger` with RRF/weighted fusion |
| **Diversity control** | `MMRHelper`, `DiversificationHelper` |
| **RAG generation** | `RAGHelper` with prompt templates |

This module exists to:

- **Ensure consistency**: Same config loading, same embedder creation across all features
- **Reduce duplication**: One implementation, reused across 17+ feature modules
- **Centralize fixes**: Bug fixes apply to all pipelines automatically
- **Standardize interfaces**: Common types and behaviors

## 3. ConfigLoader Utility

### Purpose

Config loading with environment variable resolution and validation.

### Usage

```python
from vectordb.langchain.utils import ConfigLoader

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

## 4. EmbedderHelper Utility

### Purpose

Dense embedder creation with model aliases and warm-up.

### Usage

```python
from vectordb.langchain.utils import EmbedderHelper
from langchain_huggingface import HuggingFaceEmbeddings

# Create document embedder
doc_embedder = EmbedderHelper.create_embedder(
    model="minilm",  # Alias resolves to full HF path
    device="cpu",
    batch_size=32,
)

# Create query embedder
query_embedder = EmbedderHelper.create_embedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
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

## 5. SparseEmbedder Utility

### Purpose

Sparse embedding creation for hybrid/sparse retrieval.

### Usage

```python
from vectordb.langchain.utils import SparseEmbedder

# Create sparse embedder
sparse_embedder = SparseEmbedder(
    model="naver/splade-cocondenser-ensembledistil",
    max_length=512,
)

# Embed documents
sparse_docs = sparse_embedder.embed_documents(documents)

# Embed query
sparse_query = sparse_embedder.embed_query(query)
```

### Sparse Models

| Model | Description |
|-------|-------------|
| `naver/splade-cocondenser-ensembledistil` | SPLADE ensemble (recommended) |
| `naver/splade-cocondenser-selfdistil` | SPLADE self-distilled |
| `prithvida/Splade_v2_Distilbert_uncased` | SPLADE v2 variant |

## 6. RerankerHelper Utility

### Purpose

Cross-encoder reranker creation with strategy selection.

### Usage

```python
from vectordb.langchain.utils import RerankerHelper
from langchain_core.documents import Document

# Create reranker
reranker = RerankerHelper.create_reranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=10,
)

# Run reranker
docs = [
    Document(page_content="A", metadata={"score": 0.8}),
    Document(page_content="B", metadata={"score": 0.6}),
]
reranked = reranker.rerank(query="query", documents=docs)
```

### Supported Reranker Types

| Type | Model |
|------|-------|
| **Cross-encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Cross-encoder light** | `cross-encoder/ms-marco-TinyBERT-L-2-v2` |
| **Cross-encoder stsb** | `cross-encoder/stsb-roberta-base` |

### Behavior

- **Model warm-up**: Rerankers warmed up on creation
- **Top-k truncation**: Results truncated to top_k
- **Score normalization**: Scores normalized to [0,1]

## 7. MMRHelper Utility

### Purpose

MMR (Maximal Marginal Relevance) diversity algorithm implementation.

### Usage

```python
from vectordb.langchain.utils import MMRHelper
from langchain_core.documents import Document

# MMR reranking
docs = [
    Document(page_content="A", metadata={"score": 0.9}),
    Document(page_content="B", metadata={"score": 0.8}),
    Document(page_content="C", metadata={"score": 0.7}),
]

reranked = MMRHelper.mmr_rerank(
    documents=docs,
    query_embedding=query_embedding,
    top_k=5,
    lambda_param=0.5,  # Balance relevance vs diversity
)
```

### MMR Formula

```
MMR = λ × relevance - (1-λ) × redundancy
```

| Lambda | Behavior |
|--------|----------|
| **1.0** | Pure relevance (no diversity) |
| **0.5** | Balanced relevance and diversity |
| **0.0** | Pure diversity (no relevance) |

## 8. ResultMerger Utility

### Purpose

RRF and weighted fusion utilities for hybrid retrieval.

### Usage (Stateless)

```python
from vectordb.langchain.utils import ResultMerger
from langchain_core.documents import Document

# RRF Fusion
dense_docs = [Document(page_content="A", metadata={"score": 0.8})]
sparse_docs = [Document(page_content="A", metadata={"score": 12.0}), Document(page_content="B", metadata={"score": 9.5})]

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
| **stable_doc_id(doc)** | Generate consistent IDs | Precedence: `metadata["doc_id"]` → `doc.id` → SHA1(content) |
| **rrf_fusion(dense, sparse, k, top_k)** | Reciprocal Rank Fusion | Default k=60 |
| **rrf_fusion_many(ranked_lists, k, top_k)** | N-way RRF | For multiple retrievers |
| **weighted_fusion(dense, sparse, weights, top_k)** | Weighted score fusion | Auto-normalizes weights |
| **deduplicate_by_content(docs, threshold)** | Exact hash dedup | Not semantic similarity |

### Behavior

- **Empty inputs**: Returns empty list
- **Duplicate handling**: Merges via stable IDs
- **Weight normalization**: Auto-normalizes if sum ≠ 1.0
- **Min-max normalization**: Normalizes scores to [0,1]

## 9. RAGHelper Utility

### Purpose

Generator construction and prompt formatting for answer generation.

### Usage

```python
from vectordb.langchain.utils import RAGHelper
from langchain_core.documents import Document

# Create RAG helper
helper = RAGHelper(
    model="llama-3.3-70b-versatile",
    api_key="${GROQ_API_KEY}",
    api_base_url="https://api.groq.com/openai/v1",
    temperature=0.7,
    max_tokens=2048,
)

# Format prompt
docs = [Document(page_content="Context 1"), Document(page_content="Context 2")]
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

## 10. DocumentConverter Utility

### Purpose

LangChain `Document` ↔ backend format conversion.

### Usage

```python
from vectordb.langchain.utils import DocumentConverter
from langchain_core.documents import Document

# LangChain to backend format
backend_doc = DocumentConverter.to_backend(langchain_doc)

# Backend to LangChain format
langchain_doc = DocumentConverter.from_backend(backend_doc)
```

### Behavior

- **Metadata preservation**: Maintains metadata across conversions
- **Content mapping**: `page_content` ↔ `content`
- **Score handling**: Preserves relevance scores

## 11. FiltersHelper Utility

### Purpose

MongoDB-style filter translation to backend-native formats.

### Usage

```python
from vectordb.langchain.utils import FiltersHelper

# MongoDB-style filter
filter_dict = {
    "$and": [
        {"category": {"$eq": "tech"}},
        {"year": {"$gte": 2020}},
    ]
}

# Convert to backend-native format
milvus_filter = FiltersHelper.to_milvus(filter_dict)
qdrant_filter = FiltersHelper.to_qdrant(filter_dict)
pinecone_filter = FiltersHelper.to_pinecone(filter_dict)
weaviate_filter = FiltersHelper.to_weaviate(filter_dict)
chroma_filter = FiltersHelper.to_chroma(filter_dict)
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
| **Milvus** | Boolean expression string |
| **Qdrant** | `Filter(must=[...])` |
| **Pinecone** | MongoDB-style dict |
| **Weaviate** | `Filter.by_property(...).<op>(...)` |
| **Chroma** | `where` dict |

## 12. DiversificationHelper Utility

### Purpose

Redundancy control with cosine similarity-based filtering.

### Usage

```python
from vectordb.langchain.utils import DiversificationHelper
from langchain_core.documents import Document

helper = DiversificationHelper(
    diversity_threshold=0.7,
    max_similar_docs=3,
    semantic_diversification=True,
)

docs = [
    Document(page_content="A", metadata={"score": 0.9}),
    Document(page_content="B", metadata={"score": 0.8}),
    Document(page_content="C", metadata={"score": 0.7}),
]

diverse = helper.select_diverse(docs, top_k=5)
```

### Configuration

| Parameter | Default | Impact |
|-----------|---------|--------|
| **diversity_threshold** | 0.7 | Similarity threshold for diversity |
| **max_similar_docs** | 3 | Max similar docs to keep |
| **semantic_diversification** | True | Use embeddings for diversity |

### Behavior

- **Threshold enforcement**: Removes docs above similarity threshold
- **Semantic diversity**: Uses embeddings when enabled

## 13. When to Use Utilities

### ConfigLoader

Use when:

- Loading pipeline configurations
- Need env var resolution
- Validating config structure

### EmbedderHelper

Use when:

- Creating document/query embedders
- Need model alias resolution
- Consistent embedder initialization

### SparseEmbedder

Use when:

- Creating sparse embeddings
- Hybrid retrieval needed
- SPLADE model required

### RerankerHelper

Use when:

- Creating cross-encoder rerankers
- Strategy selection needed
- Consistent ranker initialization

### MMRHelper

Use when:

- MMR diversity reranking needed
- Pure Python implementation preferred
- Lambda parameter tuning required

### ResultMerger

Use when:

- Fusing dense+sparse results
- Multiple retriever outputs need merging
- Deduplication across sources

### RAGHelper

Use when:

- Generating answers from context
- Prompt formatting needed
- OpenAI-compatible generation

### DocumentConverter

Use when:

- Converting between LangChain and backend formats
- Metadata preservation required
- Score handling needed

### FiltersHelper

Use when:

- Normalizing filter predicates
- Backend filter adaptation needed
- MongoDB-style filters required

### DiversificationHelper

Use when:

- Controlling result redundancy
- Cosine similarity-based diversity needed
- Threshold-based filtering required

## 14. When Not to Use Utilities

### Avoid When

| Utility | Avoid When |
|---------|------------|
| **ConfigLoader** | Custom config format needed |
| **EmbedderHelper** | Custom embedder logic required |
| **SparseEmbedder** | Dense-only retrieval |
| **RerankerHelper** | Custom ranking logic needed |
| **ResultMerger** | Single retriever, no fusion needed |
| **MMRHelper** | Diversity not required |
| **RAGHelper** | Custom generation pipeline |
| **DocumentConverter** | Native format sufficient |
| **FiltersHelper** | Backend-native filters sufficient |
| **DiversificationHelper** | Diversity not required |

## 15. Failure Modes and Edge Cases

### ConfigLoader

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing config file** | `FileNotFoundError` | Verify path |
| **Malformed YAML** | Parse error | Validate YAML |
| **Missing env var** | Empty string or default | Use `${VAR:-default}` |

### EmbedderHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing API key** | Raises `ValueError` | Set env var |
| **Invalid model** | Model load failure | Verify model name |
| **Device unavailable** | Falls back to CPU | Check device |

### SparseEmbedder

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Invalid model** | Model load failure | Verify HuggingFace model |
| **OOM error** | Use smaller model | Select lighter model |

### RerankerHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Model load failure** | Raises error | Verify model |
| **OOM error** | Use smaller model | Select lighter model |

### MMRHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **lambda_param outside 0-1** | Invalid results | Validate range |
| **Empty documents** | Returns empty list | Not an error |

### ResultMerger

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Empty inputs** | Returns empty list | Not an error |
| **Duplicate content** | Merges via stable IDs | Expected behavior |

### RAGHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Missing API key** | Raises `ValueError` | Set GROQ_API_KEY |
| **Generation error** | Returns None | Check API status |

### FiltersHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **Invalid operator** | Python fallback | Use supported operators |
| **Backend filter error** | Python fallback | Accept fallback |

### DiversificationHelper

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| **No diverse docs** | Returns available | Accept partial |
| **Embedding failure** | Falls back to non-semantic | Check embeddings |

## 16. Practical Usage Examples

### Example 1: Complete Utility Pipeline

```python
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    FiltersHelper,
    RerankerHelper,
    ResultMerger,
    RAGHelper,
)

# Load config
config = ConfigLoader.load("config.yaml")

# Create embedders
doc_embedder = EmbedderHelper.create_embedder(
    model=config["embeddings"]["model"],
    device=config["embeddings"]["device"],
)

# Create filter
filter_dict = {"category": {"$eq": "tech"}}
milvus_filter = FiltersHelper.to_milvus(filter_dict)

# Create ranker
ranker = RerankerHelper.create_reranker(
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
filtered = apply_filter(docs, milvus_filter)
reranked = ranker.rerank(query="query", documents=filtered)
merged = ResultMerger.rrf_fusion(reranked, [], top_k=10)
answer = rag_helper.generate(rag_helper.format_prompt("query", merged))
```

### Example 2: Hybrid Retrieval Fusion

```python
from vectordb.langchain.utils import ResultMerger
from langchain_core.documents import Document

# Dense and sparse results
dense_results = [
    {"page_content": "A", "metadata": {"score": 0.8}},
    {"page_content": "B", "metadata": {"score": 0.6}},
]
sparse_results = [
    {"page_content": "A", "metadata": {"score": 12.0}},
    {"page_content": "C", "metadata": {"score": 9.5}},
]

# Convert to Documents
dense_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in dense_results]
sparse_docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in sparse_results]

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

## 17. Source Walkthrough Map

### Primary Module Files

| File | Purpose |
|------|---------|
| `src/vectordb/langchain/utils/__init__.py` | Public API exports |
| `src/vectordb/langchain/utils/README.md` | Feature overview |

### Utility Implementations

| File | Utility |
|------|---------|
| `config.py` | `ConfigLoader` (re-exports from vectordb.utils) |
| `embeddings.py` | `EmbedderHelper` |
| `sparse_embeddings.py` | `SparseEmbedder` |
| `reranker.py` | `RerankerHelper` |
| `mmr.py` | `MMRHelper` |
| `fusion.py` | `ResultMerger` |
| `rag.py` | `RAGHelper` |
| `document_converter.py` | `DocumentConverter` |
| `filters.py` | `FiltersHelper` |
| `diversification.py` | `DiversificationHelper` |

### Test Files

| File | Coverage |
|------|----------|
| `tests/langchain/utils/test_embeddings.py` | EmbedderHelper tests |
| `tests/langchain/utils/test_sparse_embeddings.py` | SparseEmbedder tests |
| `tests/langchain/utils/test_reranker.py` | RerankerHelper tests |
| `tests/langchain/utils/test_mmr.py` | MMRHelper tests |
| `tests/langchain/utils/test_fusion.py` | ResultMerger tests |
| `tests/langchain/utils/test_filters.py` | FiltersHelper tests |
| `tests/langchain/utils/test_diversification.py` | DiversificationHelper tests |
| `tests/langchain/utils/test_config.py` | ConfigLoader tests |
| `tests/langchain/utils/test_rag.py` | RAGHelper tests |

---

**Related Documentation**:

- **Components** (`docs/langchain/components.md`): Reusable advanced-RAG components
- **Core Shared Utils** (`docs/core/shared-utils.md`): Cross-framework utilities
- **Reference Config** (`docs/reference/config-reference.md`): Configuration key inventory
