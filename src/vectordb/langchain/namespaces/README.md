# Namespaces (LangChain)

Namespaces provide logical segmentation within a shared index or collection, allowing retrieval to be scoped to a subset of data without creating separate indexes or requiring full tenant isolation infrastructure.

## How It Works

1. **Namespace assignment at indexing**: Documents are inserted with a namespace identifier stored as a metadata field or through the backend's native partitioning mechanism.
2. **Scoped retrieval**: At query time, the namespace is specified so only documents in that partition are considered for similarity ranking.
3. **Cross-namespace retrieval** (optional): Some backends allow querying across multiple namespaces in a single call.

### Per-Backend Namespace Implementations

| File | Backend | Mechanism |
|---|---|---|
| `pinecone.py` | Pinecone | Native `namespace` parameter on upsert and query. Pinecone namespaces are first-class index partitions. |
| `chroma.py` | Chroma | Metadata-based soft namespace using `where` filter on namespace field. |
| `milvus.py` | Milvus | Partition key field routing — namespace value routes documents to physical partitions. |
| `qdrant.py` | Qdrant | Payload-based filter using the `namespace` field. |
| `weaviate.py` | Weaviate | Tenant-based namespace using Weaviate's native tenant feature. |

The shared `base.py` contains common namespace scope injection logic. The `utils/` subdirectory provides namespace helpers shared across backends.

## When to Use It

- Environment separation: index `"production"`, `"staging"`, and `"test"` data in the same index, query each independently.
- Versioned document sets: maintain `"v1"` and `"v2"` knowledge bases for A/B testing.
- Customer group segmentation where groups share an index but queries must be scoped per group.

## When Not to Use It

- Hard security isolation requirements where physical separation is mandated.
- Small single-domain corpora with no partitioning need.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher precision for scoped queries by excluding irrelevant namespaces |
| Latency | Often improved for namespace-scoped queries |
| Cost | Operational complexity grows with number of namespaces |

## Configuration

```yaml
namespace:
  name: "production"
  field: "namespace"          # Metadata field name (for filter-based backends)
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `namespace.name` | Identifies which partition to read from and write to |
| `namespace.field` | Must be consistent between indexing and search scripts |
| Cross-namespace mode | Decide upfront whether your use case requires multi-namespace aggregation |

## Common Pitfalls

- **Inconsistent namespace naming**: `"prod"` and `"production"` are different namespaces. Enforce normalized, lowercase identifiers.
- **Cross-namespace deduplication**: If the same document exists in multiple namespaces, multi-namespace queries return duplicates.
- **Treating namespaces as security boundaries**: Namespaces are query scoping, not access control. Use separate infrastructure for true security isolation.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `multi_tenancy/` for full tenant lifecycle management.
- Use `metadata_filtering/` for additional soft constraints within a single namespace.
