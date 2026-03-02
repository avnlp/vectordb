# Namespaces (Haystack)

Namespaces provide logical segmentation within a shared index or collection, allowing retrieval to be scoped to a subset of data without creating separate indexes or requiring full tenant isolation. They sit between single-index retrieval (no partitioning) and full multi-tenancy (strong lifecycle management and isolation).

## How It Works

1. **Namespace assignment at indexing**: Documents are inserted into a named partition (a namespace). The namespace identifier is stored either as a metadata field or through the backend's native partitioning mechanism.
2. **Scoped retrieval**: At query time, the namespace is specified so only documents in that partition are considered.
3. **Cross-namespace retrieval** (optional): Some backends allow querying multiple namespaces or all namespaces in a single call.

### Per-Backend Namespace Implementations

The `namespaces/` directory contains per-backend files that map the namespace concept to each backend's native mechanism:

| File | Backend | Mechanism |
|---|---|---|
| `pinecone_namespaces.py` | Pinecone | Native `namespace` parameter on upsert and query. Pinecone namespaces are first-class index partitions. |
| `chroma_namespaces.py` | Chroma | Metadata-based soft namespace using `where` filter. |
| `chroma_collections.py` | Chroma | Collection-per-namespace pattern: each namespace is a separate Chroma collection. |
| `milvus_namespaces.py` | Milvus | Partition key field routing. Documents are routed to partitions by the namespace field value. |
| `qdrant_namespaces.py` | Qdrant | Payload filter using the `namespace` metadata field. |
| `weaviate_namespaces.py` | Weaviate | Tenant-based namespacing using Weaviate's native tenant feature. |

The `types.py` file defines namespace-related type literals and the `utils/` subdirectory contains shared namespace scope helpers.

## When to Use It

- Environment separation: index `"production"`, `"staging"`, and `"test"` data in the same index but query each independently.
- Versioned document sets: maintain `"v1"` and `"v2"` of a knowledge base in the same collection for A/B testing.
- Customer group segmentation where groups can share an index but queries should be scoped by group.

## When Not to Use It

- Hard security isolation requirements where physical separation (separate indexes or infrastructure) is mandated. Namespaces provide logical, not cryptographic, isolation.
- Small single-domain corpora with no need for any partitioning.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher precision for scoped queries by eliminating irrelevant namespaces |
| Latency | Often improved for namespace-scoped queries because fewer documents are searched |
| Cost | Operational complexity increases as the number of namespaces grows |

## Configuration

```yaml
namespace:
  name: "production"            # Namespace to index into / query from
  field: "namespace"            # Metadata field name (for filter-based backends)
  cross_namespace: false        # Whether to search across all namespaces
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `namespace_strategy` | How data is partitioned — native (Pinecone) vs filter-based (Chroma, Qdrant) |
| `cross_namespace_mode` | Single namespace query vs multi-namespace aggregation |
| `default_namespace` | Safe fallback behavior when namespace context is absent |

## Common Pitfalls

- **Inconsistent naming conventions**: `"prod"` and `"production"` treated as different namespaces. Enforce lowercase, normalized namespace identifiers.
- **Cross-namespace aggregation without deduplication**: If the same document was indexed in multiple namespaces, queries across namespaces may return duplicates.
- **Confusing logical isolation with hard security boundaries**: Namespaces are a query scoping mechanism, not an access control mechanism. Use separate infrastructure for true security isolation.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `multi_tenancy/` when you need full tenant lifecycle management (creation, deletion, access control).
- Use `metadata_filtering/` to apply additional constraints within a single namespace.
