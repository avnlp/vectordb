# Reranking (LangChain)

Reranking is a two-stage retrieval pattern. A fast first-pass retriever returns a broad candidate pool, and a cross-encoder model then scores each query-document pair jointly to produce a more precise final ranking.

## How It Works

1. **First-pass retrieval**: The semantic search pipeline retrieves a large candidate pool (typically 20–50 documents) using fast ANN search over precomputed embeddings.
2. **Cross-encoder reranking**: The candidate pool is passed to `RerankerHelper` (from `utils/reranker.py`), which creates a `HuggingFaceCrossEncoder` and scores each `(query, document)` pair. Cross-encoders run the query and document through the same model simultaneously, enabling fine-grained attention over query-document interactions.
3. **Top-k selection**: The reranked list is sorted by cross-encoder score and truncated to the final `top_k`. This smaller, higher-precision set is passed to the generator.

`RerankerHelper.create_reranker(config)` instantiates the cross-encoder from the config. `RerankerHelper.rerank(reranker, query, documents, top_k)` applies reranking and returns sorted documents. `rerank_with_scores()` returns `(Document, score)` tuples when scores are needed downstream.

Recommended models:
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: Fast, English-focused.
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: More accurate, slower.
- `BAAI/bge-reranker-v2-m3`: High accuracy, multilingual.

## When to Use It

- High-accuracy QA where top-1 or top-3 precision is the key metric.
- Cases where first-pass recall is acceptable but ranking quality is weak.
- Pipelines where latency allows an extra 100–300 ms for reranking.

## When Not to Use It

- Ultra-low-latency APIs where additional model inference is unacceptable.
- Very large candidate pools (100+ documents) where cross-encoder inference cost is prohibitive.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Often the largest single precision gain in the pipeline |
| Latency | Higher than single-stage retrieval; scales linearly with candidate pool size |
| Cost | Cross-encoder inference is more expensive per document than ANN search |

## Configuration

```yaml
search:
  candidate_pool_size: 20    # First-pass retrieval breadth
  top_k: 5                   # Final result count after reranking

reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Required
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `reranker.model` | Primary quality and cost tradeoff; larger models score better but slower |
| `candidate_pool_size` | Too small = missing evidence; too large = slow reranking. Set to 2–5× final `top_k`. |
| `search.top_k` | Final context size sent to the generator |

## Common Pitfalls

- **Reranking too few candidates**: If the relevant document is not in the first-pass candidate pool, reranking cannot help. Set `candidate_pool_size` generously (at least 15–25).
- **Weak first-pass retrieval**: Reranking reorders candidates but cannot add new ones. Fix retrieval recall first.
- **Not measuring latency impact**: Benchmark reranking latency on representative query loads before committing to production.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Combine with `hybrid_indexing/` for stronger first-pass recall feeding better candidates to the reranker.
- Use `contextual_compression/` after reranking to further shorten the final context.
- Use `mmr/` when result diversity matters more than maximum precision.
