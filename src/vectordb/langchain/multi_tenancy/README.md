# Multi-Tenancy (LangChain)

Multi-tenancy ensures that documents indexed for one tenant are completely invisible to queries made in another tenant's context. It is the data isolation mechanism for shared-infrastructure RAG systems serving multiple customers or business units.

## How It Works

Tenant isolation is implemented differently per backend but the pattern is consistent:

1. **Tenant context at indexing**: A tenant identifier is attached to each document during insertion. The `inject_scope_to_metadata()` utility adds `"tenant_id": "tenant_a"` (or the configured scope field) to every document's metadata.
2. **Tenant filtering at query time**: Every retrieval call automatically applies a tenant isolation filter. `inject_scope_to_filter()` injects the tenant condition into the search filter using `$and` + `$eq` logic.

### Per-Backend Isolation Mechanisms

| File | Backend | Isolation Mechanism |
|---|---|---|
| `chroma.py` | Chroma | Separate Chroma `tenant`/`database` context per tenant via `ChromaVectorDB.with_tenant()` |
| `milvus.py` | Milvus | Partition key field routing — documents route to partitions by `tenant_id` value |
| `pinecone.py` | Pinecone | Namespace-based isolation — each tenant gets its own namespace |
| `qdrant.py` | Qdrant | `is_tenant` payload index with `tenant_id` metadata field for optimized filtering |
| `weaviate.py` | Weaviate | Native Weaviate tenant context via `collection.with_tenant(tenant_name)` |

The shared `base.py` contains common multi-tenancy logic. The per-backend files handle backend-specific isolation details.

## When to Use It

- SaaS products where multiple customers share RAG infrastructure but must not see each other's data.
- Internal platforms where different business units have separate document sets.
- Any scenario with a contractual or regulatory requirement for data isolation.

## When Not to Use It

- Single-tenant deployments where isolation provides no value.
- Early prototypes where data isolation adds unnecessary configuration overhead.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Improves result correctness by preventing cross-tenant contamination |
| Latency | Usually neutral; native-tenant backends (Weaviate, Milvus partition keys) can improve query speed |
| Cost | Operational overhead for tenant lifecycle management |

## Configuration

```yaml
multi_tenancy:
  tenant_id: "tenant_a"
  scope_field: "tenant_id"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `tenant_id` propagation | Missing tenant context in any stage allows cross-tenant data leakage |
| `scope_field` | Must match the field name used during indexing for filter conditions to work |
| Native vs soft isolation | Choose the most performant isolation mechanism for your backend |

## Common Pitfalls

- **Missing tenant context in any pipeline stage**: Both indexing and search must use the same scope injection utilities. Verify the filter is applied in every search call.
- **Using soft metadata filters instead of native primitives**: On backends supporting native tenant isolation (Weaviate, Milvus), soft filtering is less efficient and potentially less reliable.
- **No tenant lifecycle policy**: Define how new tenants are provisioned and how departed tenants' data is cleaned up.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `namespaces/` for lighter logical partitioning without full tenant lifecycle management.
- Combine with `metadata_filtering/` for additional per-tenant constraints.
