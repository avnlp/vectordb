# Metadata Filtering (Haystack)

Metadata filtering applies structured constraints to retrieval so that only documents matching specific field conditions are considered. This enables scope control alongside semantic similarity scoring.

## How It Works

1. **Indexed metadata**: During indexing, each document is stored with structured metadata fields — such as `category`, `date`, `source`, `language`, or any domain-specific attribute from the dataset.
2. **Filter expression at query time**: A filter expression (dict or backend-native format) is attached to the search call. The backend applies this filter before or alongside the ANN similarity search.
3. **Scoped retrieval**: Only documents that satisfy the filter conditions enter the similarity ranking. The final result is the top-k most similar documents within the filtered set.
4. **Backend translation**: Filter conditions are expressed in a MongoDB-style dict format (`{"field": {"$eq": value}, "other_field": {"$in": [...]}}`) and translated by each backend wrapper or the `FiltersHelper` to the native filter syntax.

Each backend has its own filter implementation. Chroma uses `where` dicts, Milvus uses boolean expression strings (`metadata["field"] == "value"`), Pinecone uses `$eq`/`$in` JSON filters, Qdrant uses `Filter` + `FieldCondition` objects, and Weaviate uses its own `Filter` DSL.

## When to Use It

- Multi-domain corpora where searches must be scoped to a specific domain, category, or tenant.
- Time-windowed retrieval where only recent documents are relevant.
- Entity-constrained questions where results must mention a specific named entity stored in metadata.
- Policy or compliance constraints that mandate data scoping by source, classification, or jurisdiction.

## When Not to Use It

- Metadata that is sparsely populated or inconsistently normalized across documents. Filters on unreliable metadata produce unpredictably scoped results.
- Over-constrained filter combinations in early experiments. Start with one filter condition, verify it works, then add more.
- Queries where the user intent cannot be expressed as a structured constraint on available metadata fields.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher precision when metadata is reliable and consistently populated |
| Latency | Often similar or better than unfiltered search because the search space is smaller |
| Cost | Usually neutral; smaller candidate sets can reduce downstream generation cost |

## Filter Syntax (Common Dict Format)

```python
# Equality
{"category": {"$eq": "science"}}

# Range
{"year": {"$gt": 2020}}

# Set membership
{"language": {"$in": ["en", "fr"]}}

# Compound (AND)
{"$and": [{"category": {"$eq": "news"}}, {"year": {"$gte": 2023}}]}
```

## Configuration

```yaml
search:
  top_k: 10
  filters:
    category:
      "$eq": "science"
    year:
      "$gte": 2020
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `filters` | The main control over retrieval scope; incorrect filters eliminate all relevant results |
| `search.top_k` | Balance precision vs recall inside the filtered set; increase if filter makes the set sparse |
| Metadata schema | Consistent field names and value types across documents are the foundation of reliable filtering |

## Per-Backend Notes

Each backend in `metadata_filtering/` has its own indexing and search scripts that account for backend-specific filter syntax and indexing requirements:

- **Chroma** (`chroma.py`): Uses Chroma `where` dicts with `$eq`, `$in`, `$contains` operators.
- **Milvus** (`milvus.py`): Uses boolean expression strings; metadata fields are accessed via `metadata["field"]` JSON path syntax.
- **Pinecone** (`pinecone.py`): Uses Pinecone's JSON filter format with `$eq`, `$ne`, `$in`, `$gt`, `$lt`, `$gte`, `$lte`.
- **Qdrant** (`qdrant.py`): Uses `Filter` + `FieldCondition` objects built by `QdrantVectorDB._build_filter()`.
- **Weaviate** (`weaviate.py`): Uses Weaviate v4 `Filter` class with property conditions.

## Common Pitfalls

- **Filter syntax mismatch**: Using Pinecone's `$in` syntax for a Chroma query (or vice versa) causes silent failures or errors. Always use the wrapper's filter builder methods rather than raw backend syntax.
- **Missing metadata normalization at ingest**: If some documents have `"category": "Science"` and others have `"category": "science"`, equality filters will miss one group.
- **Combining too many hard filters**: Each additional filter condition shrinks the candidate pool. Start with the single most important constraint and verify the retrieved count is non-zero.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `multi_tenancy/` for hard isolation requirements with tenant lifecycle management.
- Use `namespaces/` for lighter logical partitioning without per-field filter syntax.
- Combine with `reranking/` to further improve ranking quality within the filtered set.
