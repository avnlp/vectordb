# Parent Document Retrieval (Haystack)

Parent document retrieval indexes small child chunks for precise semantic matching but returns the larger parent context for answer generation. This balances retrieval precision (achieved with focused small chunks) and generation coherence (achieved with rich parent context).

## How It Works

1. **Split into parents and children**: Each source document is split into large parent chunks (typically 512–2048 tokens). Each parent is then further split into smaller child chunks (typically 64–256 tokens). The child chunks are linked to their parent by a stored parent ID.
2. **Index children**: Only the child chunks are embedded and stored in the vector database. Their metadata includes `parent_id` pointing to the parent document.
3. **Retrieve children**: At query time, the query embedding is compared to child chunk embeddings. The top-k most similar children are retrieved.
4. **Resolve parents**: Retrieved children are mapped back to their parent documents using `parent_id`. Duplicate parents (when multiple children from the same parent are retrieved) are deduplicated.
5. **Return parent context**: The full text of the resolved parent documents is returned to the generator, providing coherent, complete context.

The `utils/` subdirectory contains parent-child mapping helpers. The `indexing/` scripts handle both the splitting and dual-storage logic, and the `search/` scripts handle the child-to-parent resolution step.

## When to Use It

- Long source documents (articles, reports, books) where semantic matching on the full document is too coarse but chunk-only context is too fragmented for coherent answers.
- Cases where precise retrieval matters (small child chunks are semantically focused) but coherent generation also matters (larger parents provide narrative context).
- Multi-hop questions that need several paragraphs of context from the same source document.

## When Not to Use It

- Corpora of short, self-contained documents where splitting adds no retrieval benefit.
- Pipelines with strict memory constraints where returning multiple large parent documents would exceed the generator's context window.
- Situations where child-to-parent ID mapping is unreliable (for example, dynamically generated documents without stable IDs).

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Better coherence and context completeness for generated answers |
| Latency | Moderate overhead from parent document resolution step after retrieval |
| Cost | Potentially higher generation token usage if parent chunks are large |

## Configuration

```yaml
indexing:
  parent_chunk_size: 512      # Token size of parent chunks stored for generation
  child_chunk_size: 128       # Token size of child chunks stored for retrieval
  chunk_overlap: 20           # Overlap between consecutive chunks

search:
  top_k: 5                    # Number of child chunks to retrieve
  max_parent_docs: 3          # Maximum unique parent documents to return
  retrieval_mode: "with_parents"  # "with_parents", "children_only", or "context_window"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `child_chunk_size` | Smaller chunks improve retrieval precision by focusing the embedding signal |
| `parent_chunk_size` | Larger parents provide more context but consume more tokens per generation call |
| `max_parent_docs` | Controls downstream token usage when multiple children map to the same parent |

## Common Pitfalls

- **Parent chunks too large**: If parent chunks exceed the generator's context window, the returned context must be truncated, losing the benefit of parent resolution.
- **Weak child-parent ID linking**: If parent IDs are not consistently stored and retrieved, the mapping step fails silently and the pipeline degrades to child-only retrieval.
- **No deduplication when multiple children map to the same parent**: Without deduplication, the same parent document appears multiple times in the returned context, wasting token budget.

## Output Structure

The `RetrievedDocument` dataclass (from `vectordb.utils.output`) includes a `matched_children` field that lists the specific child chunks that matched the query within each returned parent document. This is useful for highlighting which parts of the parent were responsible for its retrieval.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `semantic_search/` as the baseline when documents are already short and self-contained.
- Combine with `contextual_compression/` to shorten retrieved parent context if it still exceeds token budgets.
- Use `reranking/` on the retrieved children before parent resolution to improve which parents are selected.
