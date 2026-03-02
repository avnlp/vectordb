# Utils (Haystack)

Shared helper factories and wrappers used across all Haystack feature pipelines. These utilities centralize repeated operations so that individual feature implementations stay concise and consistent.

## Modules

### `EmbedderFactory` (`embeddings.py`)

Factory for creating and warming up Haystack SentenceTransformers embedder components.

All embedders are warmed up (`embedder.warm_up()`) immediately after creation to pre-load model weights and avoid cold-start latency on the first real request.

| Method | Returns | Use Case |
|---|---|---|
| `create_document_embedder(config)` | `SentenceTransformersDocumentEmbedder` | Batch embedding of documents during indexing |
| `create_text_embedder(config)` | `SentenceTransformersTextEmbedder` | Single-query embedding at search time |
| `create_sparse_document_embedder(config)` | `SentenceTransformersSparseDocumentEmbedder` | Sparse document embedding for hybrid indexing |
| `create_sparse_text_embedder(config)` | `SentenceTransformersSparseTextEmbedder` | Sparse query embedding for hybrid search |
| `get_embedding_dimension(embedder)` | `int` | Probe the output dimension by running a sample document |

Config structure for dense embedders:

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Required
  device: "cpu"                                       # Optional
  batch_size: 32                                      # Optional (document embedder only)
```

Config structure for sparse embedders:

```yaml
sparse:
  model: "naver/splade-cocondenser-ensembledistil"  # Required
```

```python
from vectordb.haystack.utils import EmbedderFactory

config = {"embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
doc_embedder = EmbedderFactory.create_document_embedder(config)
text_embedder = EmbedderFactory.create_text_embedder(config)
dim = EmbedderFactory.get_embedding_dimension(doc_embedder)  # e.g., 384
```

---

### `RerankerFactory` (`reranker.py`)

Factory for creating and warming up Haystack ranker components.

| Method | Returns | Use Case |
|---|---|---|
| `create(config)` | `SentenceTransformersSimilarityRanker` | Cross-encoder reranking for precision improvement |
| `create_diversity_ranker(config)` | `SentenceTransformersDiversityRanker` | MMR-based diversity-aware reranking |

Config structure for similarity ranker:

```yaml
reranker:
  model: "BAAI/bge-reranker-v2-m3"  # Required
  top_k: 5                            # Optional, default 5
```

Config structure for diversity ranker:

```yaml
mmr:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Required
  top_k: 10                                          # Optional
```

The diversity ranker always uses `strategy="maximum_margin_relevance"`.

---

### `ResultMerger` (`fusion.py`)

Score fusion for combining dense and sparse retrieval results in hybrid pipelines.

| Method | Description |
|---|---|
| `fuse_rrf(dense, sparse, top_k, k=60)` | Reciprocal Rank Fusion — robust, no parameter tuning needed |
| `fuse_weighted(dense, sparse, top_k, dense_weight=0.7, sparse_weight=0.3)` | Weighted inverse-rank fusion for explicit control |
| `fuse(dense, sparse, top_k, strategy="rrf", **kwargs)` | Unified interface selecting RRF or weighted by `strategy` argument |

Both methods deduplicate by document ID (falling back to content prefix) and return up to `top_k` results sorted by fused score.

```python
from vectordb.haystack.utils import ResultMerger

fused = ResultMerger.fuse_rrf(dense_results, sparse_results, top_k=10)
```

---

### `DiversificationHelper` (`diversification.py`)

Sequential cosine similarity count filter. Processes documents in their original relevance order and drops any document that is too similar to too many already-selected documents.

The config section:

```yaml
semantic_diversification:
  enabled: true
  diversity_threshold: 0.7   # Cosine similarity above which docs are "similar"
  max_similar_docs: 2        # Drop candidate if more than this many selected docs are similar
```

Unlike MMR, this utility does not globally optimize a relevance-diversity objective. It applies a local per-document rule while preserving original ranking order.

```python
from vectordb.haystack.utils import DiversificationHelper

filtered = DiversificationHelper.apply(documents, config)
```

---

### `RAGHelper` (`rag.py`)

Helper for LLM-based answer generation from retrieved Haystack documents.

| Method | Description |
|---|---|
| `create_generator(config)` | Creates a Groq-compatible `OpenAIGenerator`; returns `None` if `rag.enabled: false` |
| `format_prompt(query, documents, template)` | Formats retrieved documents into a numbered context prompt |
| `generate(generator, query, documents, template)` | Generates an answer and returns it as a string |

Config structure:

```yaml
rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  api_base_url: "https://api.groq.com/openai/v1"
  temperature: 0.7
  max_tokens: 2048
```

Default prompt template asks the LLM to answer the question based on the provided context, with explicit instruction to say so if the answer cannot be found.

---

### `FiltersHelper` (`filters.py`)

Translates MongoDB-style filter dictionaries to backend-native filter objects for Haystack document stores. Used by `metadata_filtering/` and other feature pipelines that need to pass filters to Haystack retrievers.

---

### `ConfigLoader` (`config.py`)

Re-exports `ConfigLoader` from `vectordb.utils.config_loader` to provide a consistent import path for all Haystack feature pipelines.

```python
from vectordb.haystack.utils import ConfigLoader

config = ConfigLoader.load("configs/pinecone_triviaqa.yaml")
ConfigLoader.validate(config, "pinecone")
```

## Common Pitfalls

- **Copy-pasting utility logic into feature scripts**: All embedder creation, reranker creation, fusion, and generation should go through these factories. Direct instantiation in feature scripts causes inconsistency when factory defaults change.
- **Calling `warm_up()` manually after factory creation**: The factories always warm up their components. Calling `warm_up()` a second time is safe but redundant.
- **Mixing factory methods for different embedding types**: `create_document_embedder` is for batch indexing; `create_text_embedder` is for single-query search. Using the wrong one at the wrong stage will work but will be unnecessarily slow.
