# Hybrid Indexing (Haystack)

Hybrid retrieval combines dense semantic embeddings with sparse lexical embeddings to improve robustness across both natural-language queries and keyword-precise queries. When one signal is weak, the other compensates.

## How It Works

1. **Dual indexing**: Each document is embedded twice — once with a dense SentenceTransformers model (`SentenceTransformersDocumentEmbedder`) to produce a float vector capturing semantic meaning, and once with a sparse SentenceTransformers model (`SentenceTransformersSparseDocumentEmbedder`, typically a SPLADE model) to produce a token-weight sparse vector capturing lexical features. Both vectors are stored in the target backend.
2. **Dual retrieval**: At query time, the query is embedded with both the dense text embedder and the sparse text embedder to produce two query representations.
3. **Score fusion**: Results from the dense retriever and the sparse retriever are merged using `ResultMerger` (from `utils/fusion.py`). The default strategy is Reciprocal Rank Fusion (RRF), which combines rankings without requiring score normalization. A weighted fusion option is also available when one signal is known to be more reliable.
4. **Final ranking**: The fused, deduplicated result list is returned as the top-k documents.

RRF formula: `score(d) = Σ 1 / (k + rank_i)` where the sum is over all retrieval sources and `k` (default 60) smooths rank differences.

## When to Use It

- Corpora with mixed query styles: some users ask in natural language, others search with domain keywords or acronyms.
- Enterprise knowledge bases where exact product names, codes, or identifiers matter alongside conceptual questions.
- Any workload where pure semantic search misses some highly relevant documents that contain exact query terms.

## When Not to Use It

- Small datasets where the added complexity of dual indexing and fusion has negligible quality impact.
- Prototypes or early experiments where you have not yet validated whether the semantic baseline falls short.
- Backends that do not natively support sparse vector storage (in those cases, sparse retrieval requires an external step).

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Usually improves recall robustness by covering both semantic and lexical intent |
| Latency | Moderate increase from running two embedding models and two retrieval paths |
| Cost | Higher indexing and query cost due to dual embeddings and more complex search |

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `fusion.strategy` | `"rrf"` requires no tuning; `"weighted"` lets you favor dense or sparse signal |
| `fusion.dense_weight` / `fusion.sparse_weight` | Only for weighted fusion; start at 0.7/0.3 and adjust based on query type distribution |
| `sparse.model` | SPLADE model quality directly affects lexical matching behavior |
| `search.top_k` | Final merged result count; set larger than the semantic-only top_k to preserve fusion coverage |

## Common Pitfalls

- **Unbalanced fusion**: Setting one weight to near-zero effectively reverts to single-signal retrieval. Measure both retrieval paths independently before fusing.
- **Missing sparse vectors at query time**: If the indexing config uses sparse embeddings but the search config does not, the sparse retrieval path returns nothing. Keep configs consistent.
- **Not validating per-query-class behavior**: Hybrid usually helps keyword queries most and natural-language queries least. If your evaluation set is exclusively natural-language questions, the improvement over semantic search may be small.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Add `reranking/` after fusion for a further precision improvement.
- Use `sparse_indexing/` alone if keyword precision is the dominant need and semantic generalization is unhelpful.
- Use `semantic_search/` as the baseline to measure how much hybrid retrieval improves quality on your evaluation set.
