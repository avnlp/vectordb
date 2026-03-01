# Diversity Filtering (LangChain)

Diversity filtering removes near-duplicate documents from retrieved results so the context passed to the generator covers different aspects of the question rather than repeating the same information.

## How It Works

Post-retrieval filtering uses cosine similarity to identify and drop near-duplicate documents. The `helpers.py` file provides LangChain-specific diversity filtering utilities. The `utils/diversification.py` helper also provides `DiversificationHelper.apply()` for threshold-based sequential filtering:

1. Documents are processed in their original relevance order.
2. For each candidate document, count how many already-selected documents have cosine similarity ≥ `diversity_threshold`.
3. Keep the candidate only if that count is below `max_similar_docs`.
4. The output is a subset of the original list with near-duplicates removed, preserving original relevance rank.

This local rule approach differs from MMR: it does not globally optimize a relevance-diversity objective but instead applies a per-document deduplication condition while maintaining the original ranking order.

The `ContextualCompressionRetriever` from LangChain can also be used to apply LangChain's native `EmbeddingsRedundantFilter` (from `langchain.retrievers.document_compressors`) as an alternative approach to diversity filtering.

## When to Use It

- Dense corpora with many overlapping or duplicate chunks from the same source.
- Open-ended questions where the generator benefits from diverse source coverage.
- Summarization tasks where near-duplicate passages inflate the apparent breadth of retrieved context.

## When Not to Use It

- Precise fact-lookup tasks where removing near-duplicates might drop slightly different but informative variants.
- Very small `top_k` (fewer than 5) where filtering cannot operate meaningfully.
- Pipelines where first-pass retrieval is already weak — fix retrieval quality first.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Better context coverage; may slightly lower top-1 precision |
| Latency | Moderate overhead from cosine similarity computation |
| Cost | Potentially lower generation cost from better context efficiency |

## Configuration

```yaml
semantic_diversification:
  enabled: true
  diversity_threshold: 0.7    # Cosine similarity above which docs are considered similar
  max_similar_docs: 2         # Drop candidate if more than this many selected docs are similar

search:
  candidate_pool_size: 20     # Retrieve this many before filtering
  top_k: 10                   # Target count after filtering
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `diversity_threshold` | Lower = more aggressive filtering. Start at 0.7 and tune based on corpus duplication density. |
| `max_similar_docs` | Set to 1 for strict uniqueness; 2–3 for softer deduplication. |
| `candidate_pool_size` | Larger initial pool gives the filter more documents to select from. |

## Common Pitfalls

- **Over-aggressive filtering**: `diversity_threshold` too low removes documents that happen to share vocabulary but contain different information.
- **Tiny candidate pools**: Filtering 5 candidates over a threshold rarely produces meaningfully different results from standard ranking. Retrieve 20–50 candidates first.
- **Expecting filtering to fix weak retrieval**: If the base retriever rarely returns relevant documents, diversity filtering cannot add missing evidence. Fix retrieval first.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `mmr/` for a joint relevance-and-diversity optimization in a single reranking pass.
- Use `reranking/` when maximum relevance precision is the goal and redundancy is acceptable.
