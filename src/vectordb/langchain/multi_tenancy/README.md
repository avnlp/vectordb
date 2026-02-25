# Multi-Tenancy

Tenant isolation capabilities across all supported vector databases, ensuring that each tenant's data is completely separated and protected from access by other tenants. This module implements database-specific isolation strategies optimized for each backend, from namespace-based isolation in Pinecone to partition-based isolation in Milvus.

Multi-tenancy is essential for SaaS applications serving multiple customers from a shared infrastructure. Each tenant operates within their own isolated data space, preventing data leakage and allowing tenant-specific customization of indexing and search parameters.

## Overview

- Complete tenant isolation with no cross-tenant data leakage
- Database-specific strategies optimized for each backend's capabilities
- Supports thousands to millions of tenants depending on the database
- Tenant-scoped indexing, search, and management operations
- Configuration-driven through YAML files with environment variable substitution
- Consistent interface across all five databases
- Supports tenant-specific metadata and access controls

## How It Works

### Tenant Isolation Strategies

Each database implements tenant isolation using the mechanism best suited to its architecture:

**Pinecone** uses namespaces to isolate tenant data within a single index. Each tenant is assigned a unique namespace, and all operations are scoped to that namespace. This approach supports tens of thousands of tenants per index with minimal overhead.

**Weaviate** uses collections (formerly classes) to isolate tenant data. Each tenant gets a dedicated collection with its own schema and configuration. Weaviate's native multi-tenancy support provides enterprise-grade isolation with per-tenant resource controls.

**Chroma** uses collections for tenant isolation. Each tenant's documents are stored in a separate collection, providing complete separation. This works well for both persistent and in-memory Chroma deployments.

**Milvus** uses partition keys to isolate tenant data within a collection. A partition key field identifies the tenant for each document, and queries automatically filter to the requesting tenant's partition. This approach supports millions of tenants with efficient query routing.

**Qdrant** uses collections for tenant isolation, similar to Weaviate and Chroma. Each tenant has a dedicated collection with independent configuration. Qdrant's payload-based filtering can also be used for multi-tenancy in some configurations.

### Indexing Phase

The indexing pipeline creates tenant-scoped collections or partitions as needed, then loads documents from a configured dataset and indexes them into the tenant's isolated storage. Each document is tagged with the tenant identifier for databases that use partition-based isolation. The pipeline handles tenant-specific configuration such as embedding models and vector dimensions.

### Search Phase

The search pipeline executes queries within the context of a specific tenant. For namespace-based databases, the tenant's namespace is used. For partition-based databases, a filter on the tenant identifier is automatically applied. The pipeline ensures that results only include documents belonging to the requesting tenant.

### Tenant Management

The module provides utilities for creating and deleting tenant storage, listing active tenants, and managing tenant-specific settings. These operations are abstracted across databases so the same management code works regardless of the backend.

## Supported Databases

| Database | Isolation Strategy | Tenant Scale | Notes |
|----------|-------------------|--------------|-------|
| Pinecone | Namespace-based | 10,000+ per index | Native namespace parameter |
| Weaviate | Collection-based | Enterprise-grade | Native multi-tenancy with per-tenant shards |
| Chroma | Collection-based | Flexible | Each tenant in separate collection |
| Milvus | Partition key-based | Millions | Partition key with automatic filtering |
| Qdrant | Collection-based | Tiered | Collections per tenant or payload filtering |

## Configuration

Configuration is stored in YAML files organized by database and dataset. The configuration specifies the tenant isolation strategy, tenant identifier fields, and all standard pipeline parameters.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "multi-tenant-index"
  # Tenant namespace is set at runtime or per-request

multi_tenancy:
  strategy: "namespace"  # or "partition_key", "collection"
  tenant_id_field: "tenant_id"
  auto_create: true  # Auto-create tenant storage on first use

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32

indexing:
  batch_size: 100

search:
  top_k: 10

logging:
  level: "INFO"
```

## Directory Structure

```
multi_tenancy/
├── __init__.py                        # Package exports
├── base.py                            # Abstract base class for multi-tenancy pipelines
├── indexing/                          # Database-specific tenant indexing
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone namespace-based indexing
│   ├── weaviate.py                    # Weaviate collection-based indexing
│   ├── chroma.py                      # Chroma collection-based indexing
│   ├── milvus.py                      # Milvus partition-based indexing
│   └── qdrant.py                      # Qdrant collection-based indexing
├── search/                            # Database-specific tenant search
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone namespace-scoped search
│   ├── weaviate.py                    # Weaviate collection-scoped search
│   ├── chroma.py                      # Chroma collection-scoped search
│   ├── milvus.py                      # Milvus partition-scoped search
│   └── qdrant.py                      # Qdrant collection-scoped search
├── pinecone.py                        # Pinecone tenant management
├── weaviate.py                        # Weaviate tenant management
├── chroma.py                          # Chroma tenant management
├── milvus.py                          # Milvus tenant management
├── qdrant.py                          # Qdrant tenant management
└── configs/                           # YAML configs organized by database
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── weaviate_triviaqa.yaml
    └── ...                            # (25+ config files total)
```

## Related Modules

- `src/vectordb/langchain/namespaces/` - Namespace management for logical partitioning
- `src/vectordb/langchain/metadata_filtering/` - Additional filtering capabilities
- `src/vectordb/langchain/semantic_search/` - Standard search without tenant isolation
