# Parent Document Retrieval (LangChain)

Parent document retrieval indexes small child chunks for precise semantic matching but returns the larger parent document context for coherent answer generation. This balances retrieval precision with generation completeness.

## How It Works

1. **Split into parents and children**: Source documents are split into large parent chunks. Each parent is further split into smaller child chunks. Each child stores its `parent_id` as metadata.
2. **Index children**: Only child chunks are embedded and stored in the vector database. Embeddings of focused small chunks provide better retrieval precision than embeddings of long parent documents.
3. **Retrieve children**: At query time, child chunk embeddings are compared to the query embedding. The top-k most similar children are retrieved.
4. **Resolve parents**: The `parent_store.py` module manages the parent-child ID mapping. Retrieved children are de-referenced to their parent documents. Duplicate parents (when multiple children share the same parent) are deduplicated.
5. **Return parent context**: The full text of the resolved parent documents is returned, giving the generator coherent, narrative-complete context.

LangChain's `ParentDocumentRetriever` can be used as an alternative to the custom `parent_store.py` logic for some backends.

## When to Use It

- Long source documents (articles, reports) where chunk-only context is too fragmented for coherent answers.
- Multi-hop questions requiring several paragraphs of context from the same source.
- Any pipeline where retrieval precision matters (small chunks) but generation coherence also matters (full parent context).

## When Not to Use It

- Short, self-contained documents where splitting adds no retrieval benefit.
- Pipelines with strict memory constraints where returning multiple large parent documents exceeds the generator's context window.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Better coherence and context completeness for generated answers |
| Latency | Moderate overhead from parent resolution step after retrieval |
| Cost | Potentially higher generation tokens if parent chunks are large |

## Configuration

```yaml
indexing:
  parent_chunk_size: 512    # Tokens per parent chunk (stored for generation)
  child_chunk_size: 128     # Tokens per child chunk (stored for retrieval)
  chunk_overlap: 20

search:
  top_k: 5                  # Child chunks to retrieve
  max_parent_docs: 3        # Maximum unique parents to return
  retrieval_mode: "with_parents"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `child_chunk_size` | Smaller = more precise retrieval; too small may lose necessary context in the child itself |
| `parent_chunk_size` | Larger = more coherent generation context; too large exceeds context window |
| `max_parent_docs` | Controls downstream token usage when multiple children map to the same parent |

## Common Pitfalls

- **Parent chunks too large**: If parents exceed the generator's context window, they must be truncated, losing the coherence benefit.
- **Inconsistent child-parent ID linking**: If parent IDs are not consistently stored and retrieved, resolution fails silently and the pipeline degrades to child-only retrieval.
- **No deduplication for multiple children per parent**: Without deduplication, the same parent appears multiple times in the returned context, wasting token budget.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `semantic_search/` as the baseline when documents are already short and self-contained.
- Combine with `contextual_compression/` to shorten retrieved parent context if it exceeds token budgets.
- Add `reranking/` on child results before parent resolution to improve which parents are selected.
