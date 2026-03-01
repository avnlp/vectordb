# Utils (LangChain)

Shared helper classes used across all LangChain feature pipelines. These utilities centralize repeated operations so individual feature implementations stay concise and consistent.

## Modules

### `EmbedderHelper` (`embeddings.py`)

Helper for creating and using HuggingFace embedding models within LangChain pipelines.

| Method | Description |
|---|---|
| `create_embedder(config)` | Creates a `HuggingFaceEmbeddings` instance from the `embeddings` config section |
| `embed_documents(embedder, documents)` | Embeds a list of LangChain `Document` objects, returning `(documents, embeddings)` tuple |
| `embed_query(embedder, query)` | Embeds a single query string, returning a `list[float]` vector |

Config structure:

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Required
  device: "cpu"                                       # Optional: "cpu" or "cuda"
  batch_size: 32                                      # Optional
```

```python
from vectordb.langchain.utils import EmbedderHelper

config = {"embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cpu"}}
embedder = EmbedderHelper.create_embedder(config)
query_vec = EmbedderHelper.embed_query(embedder, "What is machine learning?")
```

GPU acceleration (10–50× speedup for batch operations): set `device: "cuda"` when a GPU is available.

---

### `RerankerHelper` (`reranker.py`)

Helper for reranking retrieved documents using HuggingFace cross-encoder models from `langchain-community`.

| Method | Description |
|---|---|
| `create_reranker(config)` | Creates `HuggingFaceCrossEncoder` from the `reranker.model` config field |
| `rerank(reranker, query, documents, top_k)` | Scores and sorts documents; returns sorted documents |
| `rerank_with_scores(reranker, query, documents, top_k)` | Returns `list[(Document, float)]` for when scores are needed downstream |

Config structure:

```yaml
reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default if not specified
```

```python
from vectordb.langchain.utils import RerankerHelper

reranker = RerankerHelper.create_reranker(config)
reranked = RerankerHelper.rerank(reranker, query, documents, top_k=5)
```

---

### `MMRHelper` (`mmr.py`)

Pure-Python MMR (Maximal Marginal Relevance) algorithm using NumPy cosine similarity.

| Method | Description |
|---|---|
| `cosine_similarity(embedding1, embedding2)` | Computes cosine similarity between two vectors |
| `mmr_rerank(documents, embeddings, query_embedding, lambda_param, k)` | Full MMR selection returning `list[(Document, score)]` |
| `mmr_rerank_simple(documents, embeddings, query_embedding, k, lambda_param)` | Simplified version returning only documents |

The `lambda_param` controls the relevance-diversity tradeoff (1.0 = pure relevance, 0.0 = pure diversity).

```python
from vectordb.langchain.utils import MMRHelper

reranked = MMRHelper.mmr_rerank_simple(documents, embeddings, query_embedding, k=10, lambda_param=0.5)
```

---

### `ResultMerger` (`fusion.py`)

Score fusion for combining dense and sparse retrieval results in hybrid search pipelines.

| Method | Description |
|---|---|
| `fuse_rrf(dense, sparse, top_k, k=60)` | Reciprocal Rank Fusion — rank-based, no score normalization needed |
| `fuse_weighted(dense, sparse, top_k, dense_weight, sparse_weight)` | Weighted inverse-rank fusion for explicit control |
| `fuse(dense, sparse, top_k, strategy="rrf", **kwargs)` | Unified interface |

Both methods deduplicate by document ID (falling back to content prefix) before fusion.

```python
from vectordb.langchain.utils import ResultMerger

fused = ResultMerger.fuse_rrf(dense_results, sparse_results, top_k=10)
```

---

### `RAGHelper` (`rag.py`)

Helper for LLM-based answer generation from retrieved LangChain documents.

| Method | Description |
|---|---|
| `create_llm(config)` | Creates a `ChatGroq` instance; returns `None` if `rag.enabled: false` |
| `format_prompt(query, documents, template)` | Formats documents into a numbered context prompt |
| `generate(llm, query, documents, template)` | Generates an answer using the LLM and returns it as a string |

Config structure:

```yaml
rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048
```

Default prompt: `{context}\n\nQuestion: {query}\n\nAnswer:`.

---

### `DocumentConverter` (`document_converter.py`)

Converts between LangChain `Document` objects and backend-native storage formats. Used by indexing scripts to prepare documents for upsert.

---

### `FiltersHelper` (`filters.py`)

Translates MongoDB-style filter dicts to backend-native filter syntax for LangChain vector stores. Called by feature pipelines that need to pass metadata filters to LangChain retrievers.

---

### `SparseEmbeddingHelper` (`sparse_embeddings.py`)

Creates and applies sparse embedding models for LangChain hybrid search pipelines. Handles SPLADE-style sparse encoding and conversion to backend-native sparse vector formats.

---

### `DiversificationHelper` (`diversification.py`)

Sequential cosine similarity count filter for near-duplicate removal. Processes documents in original order and drops candidates that are too similar to too many already-selected documents.

Config section:

```yaml
semantic_diversification:
  enabled: true
  diversity_threshold: 0.7
  max_similar_docs: 2
```

---

### `ConfigLoader` (`config.py`)

Re-exports `ConfigLoader` from `vectordb.utils.config_loader` to provide a consistent import path for LangChain feature pipelines.

```python
from vectordb.langchain.utils import ConfigLoader

config = ConfigLoader.load("configs/pinecone_triviaqa.yaml")
ConfigLoader.validate(config, "pinecone")
```

## Common Pitfalls

- **Copy-pasting utility logic into feature scripts**: All embedding creation, reranking, MMR, and generation should go through these helpers for consistency.
- **Mixing framework-specific helpers**: LangChain utilities use `langchain_core.documents.Document` and `HuggingFaceEmbeddings`. Do not mix Haystack-specific helpers (which use `haystack.Document` and `SentenceTransformersDocumentEmbedder`) into LangChain pipelines.
- **Not handling `None` from `RAGHelper.create_llm()`**: When `rag.enabled: false`, `create_llm()` returns `None`. Check before calling `generate()`.
