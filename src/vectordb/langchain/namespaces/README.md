# Namespaces

Logical data partitioning within vector database indexes, enabling separation of documents into isolated groups without requiring multiple collections or indexes. Namespaces provide a lightweight mechanism for organizing data by tenant, version, environment, or any other categorical distinction.

This module provides a unified interface for namespace management across all five supported vector databases, abstracting the different mechanisms each database uses for logical partitioning. Whether a database uses native namespaces, collections, or partitions, the interface remains consistent.

## Overview

- Logical data separation within a single index or collection
- Support for cross-namespace search when needed
- CRUD operations for namespace management: create, delete, list, query
- Database-specific implementations using each backend's optimal mechanism
- Consistent interface across all five databases
- Suitable for dev/prod separation, versioning, or tenant organization
- Configuration-driven through YAML files with environment variable substitution

## How It Works

### Namespace Strategies by Database

Each database implements namespaces using the mechanism best suited to its architecture:

**Pinecone** provides native namespace support as a core feature. Documents are upserted to a specific namespace, and queries target that namespace by default. Pinecone namespaces are lightweight and support tens of thousands per index with no performance penalty.

**Weaviate** uses collections as namespaces. Each namespace corresponds to a Weaviate collection with its own schema. While more heavyweight than Pinecone namespaces, this provides complete isolation including separate vector spaces and configurations.

**Chroma** uses collections for namespace semantics. Each namespace is a separate Chroma collection, providing full isolation of documents and metadata.

**Milvus** uses partitions for namespace semantics. A partition key field identifies the namespace for each document, and queries filter to specific partitions. This approach is efficient and supports millions of partitions per collection.

**Qdrant** uses collections for namespace separation. Each namespace is a dedicated Qdrant collection with independent configuration and optimization settings.

### Namespace Operations

The module provides standard operations for managing namespaces across all databases:

**Create Namespace** initializes a new namespace with the specified configuration. For collection-based databases, this creates a new collection. For partition-based databases, this registers the partition key.

**Delete Namespace** removes a namespace and all its documents. For collection-based databases, the collection is deleted. For partition-based databases, all documents with that partition key are removed.

**List Namespaces** returns all available namespaces in the database. The implementation handles the database-specific APIs for enumerating collections or partitions.

**Query Namespace** executes a search within a specific namespace, returning only documents from that namespace. This is the primary search operation for namespace-scoped retrieval.

**Upsert to Namespace** adds or updates documents within a specific namespace. Documents are tagged with the namespace identifier for proper isolation.

### Use Cases

Namespaces are ideal for scenarios requiring logical separation:

- **Environment separation**: Keep development, staging, and production data isolated within the same infrastructure
- **Versioning**: Maintain multiple versions of a document set for A/B testing or rollback
- **Multi-tenancy (lightweight)**: Separate tenant data when the number of tenants is moderate
- **Content organization**: Group documents by category, project, or source for easier management

## Supported Databases

| Database | Namespace Mechanism | Capacity | Notes |
|----------|-------------------|----------|-------|
| Pinecone | Native namespaces | 100,000+ per index | Zero overhead per namespace |
| Weaviate | Collections | Hundreds | Full schema isolation per namespace |
| Chroma | Collections | Hundreds | Complete document isolation |
| Milvus | Partitions | Millions | Efficient partition pruning |
| Qdrant | Collections | Hundreds | Independent optimization per namespace |

## Configuration

Configuration is stored in YAML files that define namespace settings and database connection parameters.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "namespace-index"
  # Namespace is typically set at runtime

namespaces:
  default_namespace: "default"
  auto_create: true  # Create namespace if it doesn't exist
  allowed_namespaces:  # Optional whitelist
    - "production"
    - "staging"
    - "development"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32

search:
  top_k: 10
  # Enable cross-namespace search to query multiple namespaces
  cross_namespace: false

logging:
  level: "INFO"
```

## Directory Structure

```
namespaces/
├── __init__.py                        # Package exports
├── base.py                            # Abstract NamespacePipeline class
├── pinecone.py                        # Pinecone namespace implementation
├── weaviate.py                        # Weaviate namespace implementation
├── chroma.py                          # Chroma namespace implementation
├── milvus.py                          # Milvus namespace implementation
├── qdrant.py                          # Qdrant namespace implementation
└── configs/                           # YAML configs
    └── namespace_config.yaml
```

## Related Modules

- `src/vectordb/langchain/multi_tenancy/` - Full multi-tenant isolation with per-tenant optimization
- `src/vectordb/langchain/semantic_search/` - Standard search without namespace isolation
- `src/vectordb/langchain/metadata_filtering/` - Additional filtering capabilities
