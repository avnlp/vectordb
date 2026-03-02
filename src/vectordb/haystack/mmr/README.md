# MMR (Haystack)

MMR (Maximal Marginal Relevance) is a reranking strategy that selects documents to maximize both relevance to the query and diversity from already-selected documents. It prevents the final context from being dominated by near-duplicate passages.

## How It Works

1. **Initial retrieval**: A standard first-pass retrieval returns a candidate pool (typically larger than the final `top_k` to give MMR enough candidates to choose from). This is called `fetch_k` in some configurations.
2. **Iterative MMR selection**: Documents are selected one at a time using the formula:
   - `MMR(d) = λ × sim(d, query) − (1 − λ) × max_sim(d, selected)`
   - The first term rewards relevance (cosine similarity between document embedding and query embedding).
   - The second term penalizes redundancy (maximum cosine similarity to any already-selected document).
3. **Result**: A set of documents that balances strong relevance with topical coverage.

In Haystack, MMR is implemented via `SentenceTransformersDiversityRanker` with `strategy="maximum_margin_relevance"`, created and warmed up by `RerankerFactory.create_diversity_ranker(config)`.

The `lambda` parameter (also called `lambda_threshold` in some Haystack configs) controls the tradeoff:
- `lambda = 1.0`: Pure relevance (same as standard ranking).
- `lambda = 0.5`: Equal weight on relevance and novelty (good starting default).
- `lambda = 0.0`: Maximum diversity (ignores relevance entirely; rarely useful).

## When to Use It

- Open-ended questions that may have multiple relevant subtopics in the corpus.
- Summarization and synthesis tasks where the generator benefits from diverse source coverage.
- Dense corpora where many retrieved documents are near-duplicates of the same passage.
- When the top-1 document is adequate but the remaining context slots tend to repeat the same information.

## When Not to Use It

- Fact-lookup tasks where the single most precise snippet is what matters and diversity is irrelevant.
- Very small `top_k` values (less than 3–4) where diversity cannot operate meaningfully over so few candidates.
- Ultra-low-latency paths where even the moderate overhead of embedding-based MMR selection is unacceptable.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Better context coverage across subtopics; may slightly lower top-1 precision |
| Latency | Low-to-moderate overhead compared to pure reranking; requires document embeddings |
| Cost | Neutral to slightly positive: better context breadth can reduce generator retries |

## Configuration

```yaml
mmr:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Required: model for diversity ranker
  top_k: 10          # Final result count after MMR selection
  lambda_threshold: 0.5  # Balance between relevance and diversity

search:
  fetch_k: 30        # Candidate pool size fed to MMR (should be larger than top_k)
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `mmr.lambda_threshold` (or `lambda`) | Higher favors relevance; lower favors novelty. Start at 0.5 and adjust. |
| `search.fetch_k` | The candidate pool MMR selects from. Larger pools give better diversity opportunities. |
| `mmr.top_k` | Final selected set size. Should be smaller than `fetch_k`. |

## Common Pitfalls

- **Confusing `fetch_k` with `top_k`**: `fetch_k` is the number of candidates retrieved for MMR to select from. `top_k` is the final output count. Always set `fetch_k > top_k`.
- **Using default `lambda` without evaluation**: The default may not match your domain. Evaluate on your query set before accepting the default.
- **Applying MMR to an already tiny candidate set**: MMR diversity requires enough candidates to choose differently ranked documents. Small pools produce trivially different results from standard ranking.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `diversity_filtering/` for a simpler, configurable threshold-based approach to deduplication without global MMR optimization.
- Use `reranking/` when pure relevance precision is the goal and diversity is not a concern.
- Use `contextual_compression/` to shorten the final selected context if MMR results are still too long.
