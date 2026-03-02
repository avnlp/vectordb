# JSON Indexing (Haystack)

JSON indexing is designed for document corpora where content is stored as JSON records with both textual fields (for semantic search) and structured attribute fields (for filter-based retrieval). It combines embedding-based similarity with precise field-level constraints.

## How It Works

1. **JSON document ingestion**: Raw JSON records are loaded and parsed. One or more text fields (for example, `"description"`, `"body"`, `"summary"`) are selected for embedding. The remaining structured fields become metadata for filtering.
2. **Selective embedding**: Only the designated text fields are embedded. Embedding irrelevant fields (like numeric IDs or timestamps) dilutes the embedding signal.
3. **Metadata preservation**: All structured attributes are stored as metadata alongside the embedding. Field names are normalized to avoid backend-specific conflicts.
4. **Filter-aware retrieval**: At query time, a filter expression constrains the search to documents matching structural attributes (for example, `{"status": "active", "region": "eu"}`), and similarity ranking operates within the filtered set.

JSON indexing uses `json_indexing/common/` for shared parsing and normalization logic, and `json_indexing/indexing/` and `json_indexing/search/` for per-backend implementations.

## When to Use It

- API responses or event streams stored as JSON records.
- Product catalogs, knowledge bases, or CRMs where each record has structured attributes alongside descriptive text.
- Mixed-content datasets where semantic search on the text content must be combined with exact matches on metadata attributes.
- Any corpus where the schema is well-known at indexing time and querying often uses structured constraints.

## When Not to Use It

- Pure free-text corpora with no meaningful structured attributes; standard semantic search suffices.
- Unstable or evolving schemas where the metadata fields change frequently, making filter queries unreliable.
- Deeply nested JSON with arrays of objects that require complex flattening strategies.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher precision for queries that combine semantic and structured intent |
| Latency | Moderate; depends on filter complexity and indexed metadata volume |
| Cost | Slightly higher ingest complexity for schema parsing; query cost similar to filtered semantic search |

## Configuration

```yaml
search:
  top_k: 10
  text_field: "description"     # Primary field to embed
  metadata_fields:
    - "category"
    - "region"
    - "status"
  filters:
    status:
      "$eq": "active"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `text_field_selection` | Embedding the right text field is the primary quality lever |
| `metadata_fields` | Only index fields you plan to filter on; excess fields add noise and storage overhead |
| `normalization_rules` | Consistent value normalization (lowercase, trim whitespace) ensures filters work reliably |

## Common Pitfalls

- **Embedding irrelevant JSON fields**: Including numeric IDs, timestamps, or internal codes in the embedded text dilutes the semantic signal and degrades retrieval quality.
- **Unbounded nested metadata**: Deeply nested JSON fields require flattening strategies. Use dot-notation keys (`"user.address.city"`) or explicitly select which levels to include.
- **No schema conventions across producers**: If different producers write the same field with different names or value formats, filter expressions become unreliable.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `metadata_filtering/` when documents are already normalized flat records rather than raw JSON.
- Use `semantic_search/` as a baseline when structured constraints are minimal or the schema is not yet stable.
- Add `reranking/` after JSON-filtered retrieval for additional precision.
