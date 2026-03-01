# Diversity Filtering (Haystack)

Diversity filtering removes near-duplicate documents from retrieved results so the final context covers different aspects of a question rather than repeating the same information multiple times.

## How It Works

The Haystack diversity filtering pipelines use `SentenceTransformersDiversityRanker` (from `utils/reranker.py` via `RerankerFactory.create_diversity_ranker()`) for the main diversity-aware reranking step.

A lighter auxiliary approach is available in `utils/diversification.py` via `DiversificationHelper.apply()`. This implements a sequential similarity count filter:

1. Documents are processed in their current order (preserving original relevance ranking).
2. For each candidate document, count how many already-selected documents have cosine similarity ≥ `diversity_threshold`.
3. Keep the document only if that count is less than `max_similar_docs`.
4. The result is a subset of the original list with near-duplicates removed.

This local rule preserves the original relevance order while filtering excessive redundancy, unlike MMR which globally optimizes a relevance-diversity objective.

The `DiversificationHelper` uses cosine similarity computed from document embeddings (pure Python, no GPU required for this step).

## When to Use It

- Open-ended questions with many relevant angles where a diverse context helps the generator synthesize a better answer.
- Summarization or synthesis tasks where the generator suffers from reading the same facts repeated across multiple documents.
- Dense corpora with many very similar chunks (for example, a corpus where the same article was chunked into 20 overlapping windows).

## When Not to Use It

- Precise fact-lookup tasks where the single most relevant snippet is what matters; removing near-duplicates may drop slightly different but still informative variants.
- Very small `top_k` values (3 or fewer) where diversity filtering cannot meaningfully operate.
- Pipelines where first-pass retrieval recall is already weak; fix recall before adding diversity filtering.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Better context coverage; may slightly lower top-1 precision in precision-focused tasks |
| Latency | Moderate overhead from similarity computation; scales with candidate pool size |
| Cost | Potentially lower generation cost from better context breadth reducing retries |

## Configuration

For `DiversificationHelper`:

```yaml
semantic_diversification:
  enabled: true
  diversity_threshold: 0.7    # Cosine similarity above which two docs are "similar"
  max_similar_docs: 2         # Maximum number of similar docs allowed for each new candidate
```

For `SentenceTransformersDiversityRanker` (via config):

```yaml
mmr:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 10
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `diversity_threshold` | Lower threshold = more aggressive filtering. Start at 0.7 and adjust. |
| `max_similar_docs` | Set to 1 for strict uniqueness, 2–3 for softer filtering. |
| `candidate_pool_size` | Larger initial pool gives the filter more documents to choose from. |

## Common Pitfalls

- **Over-aggressive diversity settings**: Setting `diversity_threshold` too low removes documents that are relevant but happen to share vocabulary with the first selected result.
- **Using tiny candidate pools**: Diversity filtering over 5 candidates cannot produce meaningfully different results. Retrieve 20–50 candidates and filter down.
- **Expecting diversity to fix weak base retrieval**: If the first-pass retriever rarely returns relevant documents, diversity filtering cannot help. Fix retrieval quality first.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `mmr/` for a single-pass relevance-plus-diversity ranking that avoids the two-step retrieve-then-filter pattern.
- Use `reranking/` when maximum relevance precision matters and redundancy is acceptable.
