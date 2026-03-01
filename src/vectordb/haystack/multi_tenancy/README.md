# Multi-Tenancy (Haystack)

Multi-tenancy ensures that documents indexed for one tenant are completely invisible to queries made in the context of another tenant. It is the data isolation mechanism for shared-infrastructure RAG systems serving multiple customers or business units.

## How It Works

Tenant isolation is implemented differently per backend, but the pattern is consistent:

1. **Tenant context at indexing**: During document insertion, a tenant identifier is attached to each document. The `inject_scope_to_metadata()` utility (from `vectordb.utils.scope`) adds `"tenant_id": "tenant_a"` (or the configured scope field) to every document's metadata.
2. **Tenant-aware schema**: Some backends use physical isolation at the storage layer. Others use logical isolation via metadata filters.
3. **Tenant filtering at query time**: Every search call automatically applies a tenant isolation filter so only documents matching the current tenant's ID are considered. The `inject_scope_to_filter()` utility injects the tenant condition into the query filter using `$and` + `$eq`.

### Per-Backend Isolation Mechanisms

| Backend | Isolation Mechanism |
|---|---|
| **Chroma** (`chroma/`) | Separate Chroma `tenant` and `database` context per tenant. The `ChromaVectorDB.with_tenant()` method returns a new wrapper instance scoped to the tenant. |
| **Milvus** (`milvus/`) | Partition key field (`is_partition_key=True`) routes documents to different physical partitions. Queries scoped by partition key value skip non-matching partitions entirely. |
| **Pinecone** (`pinecone/`) | Namespace-based isolation. Each tenant gets its own namespace; queries are scoped to that namespace. |
| **Qdrant** (`qdrant/`) | `is_tenant` payload index optimization with `tenant_id` metadata field. Qdrant optimizes filter performance when a field is marked as a tenant key. |
| **Weaviate** (`weaviate/`) | Native Weaviate tenants. Tenants are created with `create_tenants()`, and the collection context is switched with `with_tenant()` before upsert or query. |

## When to Use It

- SaaS products where multiple customers share the same RAG infrastructure but must not see each other's data.
- Internal platforms where different business units have separate document sets and must be kept isolated.
- Any scenario with a hard contractual or regulatory requirement that customer data not be co-mingled in queries.

## When Not to Use It

- Single-tenant deployments with no isolation requirement — the added complexity provides no value.
- Early prototypes where data isolation is not yet a concern and adds configuration overhead.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Improves result correctness by preventing cross-tenant contamination |
| Latency | Usually neutral; partition-key (Milvus) and native-tenant (Weaviate) isolation can improve query speed |
| Cost | Operational overhead for tenant lifecycle management (creation, deletion, monitoring) |

## Configuration

```yaml
multi_tenancy:
  tenant_id: "tenant_a"         # Tenant identifier for this pipeline run
  scope_field: "tenant_id"      # Metadata field name used for isolation
  isolation_mode: "partition"   # Backend-specific; e.g., "partition", "namespace", "tenant"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `tenant_id` propagation | Missing tenant context in any pipeline stage allows cross-tenant data leakage |
| `isolation_mode` | Controls which backend-native mechanism is used; choose the most performant for your backend |
| Default tenant policy | Define safe behavior when tenant context is absent (reject the query or use a default tenant) |

## Common Pitfalls

- **Missing tenant context in one pipeline stage**: If indexing stamps `tenant_id` on documents but the search script does not apply the filter, all tenants' data becomes visible. Verify both stages use the same scope injection utilities.
- **Using soft metadata filters instead of native tenant primitives**: On backends that support native tenant isolation (Weaviate, Milvus partition keys), soft filtering is less efficient and potentially less reliable than using the native mechanism.
- **No tenant onboarding policy**: Define how new tenants are provisioned (collection creation, tenant registration, initial data ingestion) and how departed tenants are cleaned up.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `namespaces/` for lighter logical partitioning without full tenant lifecycle management.
- Combine with `metadata_filtering/` to apply additional per-tenant constraints within an already isolated context.
