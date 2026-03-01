# MMR (LangChain)

MMR (Maximal Marginal Relevance) is a diversity-aware reranking strategy that selects documents maximizing both relevance to the query and novelty relative to already-selected documents.

## How It Works

The `MMRHelper` class (from `utils/mmr.py`) implements the full MMR algorithm in pure Python with NumPy:

1. **Initial retrieval**: A standard first-pass retrieval returns a large candidate pool (`fetch_k`). All candidate embeddings are collected alongside the query embedding.
2. **Greedy MMR selection**: Documents are selected iteratively:
   - Compute `relevance = cosine_similarity(document, query)` for all remaining candidates.
   - Compute `redundancy = max(cosine_similarity(document, selected_doc))` over all already-selected documents.
   - Score: `MMR(d) = λ × relevance − (1 − λ) × redundancy`.
   - Select the document with the highest MMR score and remove it from the candidate pool.
3. **Result**: A ranked list of `k` documents balancing strong relevance with minimal redundancy.

The `lambda_param` controls the tradeoff:
- `1.0`: Pure relevance (identical to standard ranking).
- `0.7–0.8`: Mild diversity penalty (recommended for precision-oriented tasks).
- `0.5`: Balanced relevance and diversity (good default for most RAG use cases).
- `0.3–0.4`: Stronger diversity emphasis (useful for exploratory search and summarization).
- `0.0`: Pure diversity (maximum spread, ignores relevance).

`MMRHelper.mmr_rerank()` returns `(Document, score)` tuples. `MMRHelper.mmr_rerank_simple()` returns just documents for simpler integration.

## When to Use It

- Open-ended questions where the corpus contains multiple relevant subtopics.
- Summarization tasks where the generator benefits from diverse source coverage.
- Dense corpora with many near-duplicate chunks from the same article.

## When Not to Use It

- Fact-lookup tasks where the single most relevant snippet is what matters.
- Very small `top_k` values (fewer than 4) where diversity cannot operate meaningfully.
- Latency-sensitive paths where embedding-based MMR computation is too expensive.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Better topical coverage; may slightly lower top-1 precision |
| Latency | Low-to-moderate overhead; requires document embeddings at MMR time |
| Cost | Neutral to slightly positive from better context breadth |

## Configuration

```yaml
mmr:
  lambda_param: 0.5     # Relevance-diversity balance
  k: 10                 # Number of documents to select
  fetch_k: 30           # Candidate pool before MMR selection

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `mmr.lambda_param` | The most important knob; tune based on whether your task prioritizes precision or coverage |
| `mmr.fetch_k` | Larger pools give MMR more options to diversify from; should be larger than `k` |
| `mmr.k` | Final selected set size; smaller = less context noise, larger = more coverage |

## Common Pitfalls

- **Confusing `fetch_k` and `k`**: `fetch_k` is the input pool size; `k` is the output size. Always set `fetch_k > k`.
- **Using default `lambda_param` without evaluation**: Different tasks benefit from different diversity levels. Evaluate against your query set.
- **Applying MMR to tiny candidate sets**: MMR over 3–4 candidates produces nearly the same result as standard ranking. Use larger pools.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `diversity_filtering/` for a simpler threshold-based deduplication approach.
- Use `reranking/` when maximum relevance precision is the goal.
- Use `contextual_compression/` to shorten the MMR-selected context if it is still too long.
