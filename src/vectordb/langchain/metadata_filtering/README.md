# Metadata Filtering (LangChain)

Metadata filtering applies structured constraints to retrieval so only documents matching specific field conditions are searched. This enables precise scope control alongside semantic similarity scoring.

## How It Works

1. **Indexed metadata**: During indexing, documents are stored with structured metadata fields. LangChain backends preserve these as metadata on each document record.
2. **Filter expression at query time**: A filter expression is attached to the LangChain retriever. The backend evaluates the filter before or alongside ANN similarity search.
3. **Scoped retrieval**: Only documents satisfying the filter conditions enter the similarity ranking. The top-k most similar documents within the filtered set are returned.
4. **Backend translation**: Filter conditions are expressed in a MongoDB-style dict format and translated to the backend's native filter syntax by `FiltersHelper` (from `utils/filters.py`).

LangChain vector stores expose filtering via their `similarity_search(query, filter=...)` or retriever `search_kwargs={"filter": ...}` interfaces. The filter format varies slightly per backend:

| Backend | Filter Format |
|---|---|
| Chroma | `{"field": {"$eq": value}}` |
| Milvus | Milvus expression string |
| Pinecone | Pinecone JSON filter (`{"$and": [...]}`) |
| Qdrant | `qdrant_client.http.models.Filter` object |
| Weaviate | Weaviate `Filter` class |

## When to Use It

- Multi-domain corpora where results must be scoped by category, topic, or source.
- Time-windowed retrieval where only recent documents are relevant.
- Entity-constrained questions where the answer must reference a specific named entity.
- Compliance constraints that mandate scoping by data classification or jurisdiction.

## When Not to Use It

- Sparse or inconsistently populated metadata. Filters on unreliable fields produce unpredictably scoped results.
- Early experiments before verifying that metadata fields are consistently normalized.
- Over-constrained filter combinations that eliminate all relevant candidates.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher precision when metadata is reliable and consistently normalized |
| Latency | Often similar or better than unfiltered search because the candidate set is smaller |
| Cost | Usually neutral; smaller filtered sets reduce downstream reranking and generation cost |

## Filter Syntax (Common Dict Format)

```python
# Equality
{"category": {"$eq": "science"}}

# Range
{"year": {"$gt": 2020}}

# Set membership
{"language": {"$in": ["en", "fr"]}}

# Combined
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
| `filters` | Main scope control; incorrect filters eliminate all relevant results |
| `search.top_k` | Increase if filtering makes the candidate set sparse |
| Metadata schema | Consistent field names and values are the foundation of reliable filtering |

## Common Pitfalls

- **Filter syntax mismatch**: Each backend uses different filter syntax. Use `FiltersHelper` to translate from the common dict format to backend-native syntax rather than writing raw backend filters in application code.
- **Missing metadata normalization**: If some documents have `"category": "Science"` and others `"category": "science"`, equality filters miss one group.
- **Combining too many hard filters**: Each additional condition shrinks the candidate pool. Start with the single most important constraint and verify non-zero results before adding more.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `multi_tenancy/` for full tenant isolation with lifecycle management.
- Use `namespaces/` for lighter logical partitioning.
- Combine with `reranking/` for higher precision within the filtered set.
