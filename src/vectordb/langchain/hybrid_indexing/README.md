# Hybrid Indexing (LangChain)

Hybrid retrieval combines dense semantic embeddings with sparse lexical embeddings to improve robustness across both natural-language and keyword-precise queries.

## How It Works

1. **Dual indexing**: Each document is embedded twice — once with `HuggingFaceEmbeddings` for the dense semantic vector, and once with a sparse embedding model for the token-weight lexical vector. Both vectors are stored in the backend.
2. **Dual retrieval**: At query time, the same query is embedded with both the dense and sparse models.
3. **Score fusion**: Dense and sparse retrieval results are merged using `ResultMerger` from `utils/fusion.py`. The default strategy is Reciprocal Rank Fusion (RRF). A weighted fusion option gives explicit control over dense vs sparse contribution.
4. **Final ranking**: The fused, deduplicated list (up to `top_k`) is returned.

`ResultMerger.fuse_rrf()` combines rankings using `score(d) = Σ 1 / (k + rank)` across all retrieval sources. This is robust to different score scales because it operates on ranks, not raw scores.

## When to Use It

- Mixed query styles where some users phrase naturally and others search with domain terms.
- Enterprise knowledge bases with exact product names, codes, or identifiers alongside conceptual questions.
- Any workload where pure semantic search misses documents containing exact query terms.

## When Not to Use It

- Small datasets where dual indexing complexity has negligible quality impact.
- Prototypes where the semantic baseline has not yet been validated.
- Backends that do not natively support sparse vectors.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Usually improves recall robustness by covering both semantic and lexical intent |
| Latency | Moderate increase from two embedding models and two retrieval paths |
| Cost | Higher indexing and query cost from dual embeddings and more complex search |

## Configuration

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"

sparse:
  model: "naver/splade-cocondenser-ensembledistil"

fusion:
  strategy: "rrf"    # "rrf" or "weighted"
  dense_weight: 0.7  # Used only when strategy is "weighted"
  sparse_weight: 0.3

search:
  top_k: 10
  fetch_k: 30        # Candidate pool per retriever before fusion
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `fusion.strategy` | `"rrf"` requires no tuning; `"weighted"` gives explicit control |
| `sparse.model` | SPLADE model quality directly affects lexical matching coverage |
| `search.fetch_k` | Each retriever fetches this many candidates before fusion; larger pools improve fusion quality |

## Common Pitfalls

- **Unbalanced fusion**: Weight near-zero on either side effectively reverts to single-signal retrieval. Measure both retrieval paths independently first.
- **Missing sparse model at query time**: Ensure both dense and sparse embedding configs are consistent between indexing and search scripts.
- **Not validating per-query-class behavior**: Hybrid helps keyword-heavy queries most. If your evaluation set is all natural-language questions, the improvement over semantic search may be modest.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Add `reranking/` after fusion for a further precision improvement on the merged result set.
- Use `sparse_indexing/` alone if keyword precision is the dominant need.
- Measure against `semantic_search/` to quantify the hybrid improvement on your evaluation set.
