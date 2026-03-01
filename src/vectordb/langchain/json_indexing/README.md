# JSON Indexing (LangChain)

JSON indexing is designed for document corpora where content is stored as JSON records with both textual fields for semantic search and structured attribute fields for filter-based retrieval.

## How It Works

1. **JSON document parsing**: Raw JSON records are loaded and parsed. Selected text fields are concatenated or extracted to form the document content for embedding. Remaining structured attributes are stored as LangChain `Document` metadata.
2. **Selective embedding**: Only the designated text field(s) are embedded using `HuggingFaceEmbeddings`. Embedding numeric IDs, timestamps, or unrelated codes dilutes the semantic signal.
3. **Structured metadata storage**: All schema fields are stored as `page_content`-adjacent metadata on the LangChain `Document`. LangChain backends preserve these for filter-based retrieval.
4. **Filter-aware retrieval**: At query time, filter conditions on metadata fields restrict the search space before similarity ranking.

The `conftest.py` file (for the json_indexing module) provides shared test fixtures for JSON document parsing and conversion.

## When to Use It

- API response data, event streams, or product catalogs stored as JSON.
- Knowledge bases where each record has structured attributes alongside descriptive text.
- Any corpus where both semantic content and structured attributes drive retrieval decisions.

## When Not to Use It

- Pure free-text corpora with no meaningful structured attributes.
- Unstable or rapidly evolving schemas that make filter expressions unreliable.
- Deeply nested JSON structures without a normalization strategy.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher precision for queries combining semantic and structured intent |
| Latency | Moderate; depends on filter complexity and metadata indexing |
| Cost | Slightly higher ingest complexity; query cost similar to filtered semantic search |

## Configuration

```yaml
search:
  top_k: 10
  text_field: "description"
  metadata_fields:
    - "category"
    - "status"
    - "region"
  filters:
    status:
      "$eq": "active"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `text_field` | Embedding the right field is the primary quality lever |
| `metadata_fields` | Only index fields you plan to filter on; extra fields add storage overhead |
| Normalization | Consistent field formats prevent silent filter mismatches |

## Common Pitfalls

- **Embedding irrelevant fields**: Including numeric IDs or timestamps in embedded text dilutes the semantic signal.
- **Unbounded nesting**: Deeply nested JSON requires explicit flattening strategy. Use dot-notation keys for nested fields (`"user.city"`).
- **No schema conventions**: Different JSON producers using different field names or value formats make filter queries unreliable.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `metadata_filtering/` when documents are already normalized flat records.
- Use `semantic_search/` as a simpler baseline when structured constraints are minimal.
- Add `reranking/` after JSON-filtered retrieval for additional ranking precision.
