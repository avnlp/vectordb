# Reranking (Haystack)

Reranking is a two-stage retrieval pattern. A fast first-pass retriever returns a broad candidate pool, and a more powerful cross-encoder model then reranks those candidates by scoring each query-document pair jointly. This produces a significantly more precise final ranking than single-stage retrieval alone.

## How It Works

1. **First-pass retrieval**: The standard semantic search pipeline retrieves a large candidate pool (typically 20–50 documents). This step is fast because it uses ANN search over precomputed embeddings.
2. **Cross-encoder reranking**: The candidate pool is passed to `SentenceTransformersSimilarityRanker`, which scores each `(query, document)` pair by running the query and document together through a cross-encoder model. Cross-encoders apply self-attention across the full query-document pair, capturing fine-grained semantic interactions that bi-encoder similarity cannot.
3. **Top-k selection**: The reranked list is truncated to the configured `top_k`. This smaller, higher-quality set is what gets passed to the generator.

The `RerankerFactory.create(config)` helper creates and warms up the `SentenceTransformersSimilarityRanker` from the config dict. Warm-up loads model weights before the first query to avoid cold-start latency.

Recommended reranker models:
- `BAAI/bge-reranker-v2-m3`: High accuracy, multilingual, moderate inference speed.
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: Faster, English-focused, good for latency-sensitive pipelines.
- `BAAI/bge-reranker-base`: Good balance of speed and accuracy.

## When to Use It

- High-accuracy QA tasks where the rank of the first relevant document matters (MRR and NDCG).
- Cases where first-pass retrieval recall is acceptable but the top-1 or top-3 precision is weak.
- Any pipeline where you can afford higher per-query latency in exchange for better ranking quality.

## When Not to Use It

- Ultra-low-latency APIs where adding even 50–200 ms of cross-encoder inference is unacceptable.
- Cost-constrained pipelines with very large candidate pools (reranking 100 candidates is expensive; use smaller `candidate_pool_size`).

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Often the single largest precision gain available in a retrieval pipeline |
| Latency | Higher than first-pass retrieval alone; scales linearly with candidate pool size |
| Cost | Cross-encoder inference is more expensive per document than ANN search |

## Configuration

```yaml
search:
  top_k: 5               # Final result count after reranking

reranker:
  model: "BAAI/bge-reranker-v2-m3"  # Required: full model path
  top_k: 5               # Matches or is less than search.top_k
  candidate_pool_size: 20  # First-pass retrieval breadth; feeds the reranker input
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `reranker.model` | Primary quality and cost tradeoff lever |
| `candidate_pool_size` (first-pass top_k) | Too small = missing evidence; too large = slow reranking |
| `reranker.top_k` | Final context size; balance between coverage and token cost |

## Common Pitfalls

- **Reranking too few candidates**: If the cross-encoder never sees the relevant document because first-pass retrieval did not retrieve it, reranking cannot recover it. Set `candidate_pool_size` to at least 2–5× your final `top_k`.
- **Weak first-pass retrieval**: Reranking can reshuffle candidates but cannot add new ones. Fix retrieval recall first (better embedding model, larger pool, or hybrid retrieval).
- **Ignoring latency in production**: Reranking 20 documents per query may add 100–300 ms depending on the model and hardware. Benchmark on representative query loads before deploying.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Combine with `hybrid_indexing/` for stronger first-pass recall feeding better candidates into the reranker.
- Use `contextual_compression/` after reranking if the top-ranked documents are still too long or noisy for the generator.
- Use `mmr/` instead of reranking when result diversity matters more than pure precision.
