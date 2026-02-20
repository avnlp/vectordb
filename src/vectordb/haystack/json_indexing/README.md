# JSON Indexing

This module provides structured JSON document indexing and filtered search across all five supported vector databases. It indexes documents with rich JSON metadata and translates user-provided filter expressions into the native query syntax of each database, enabling field-level filtering during vector similarity search.

Each database uses a different filtering mechanism internally: dictionary-based filters for Pinecone, GraphQL expressions for Weaviate, boolean expressions for Milvus, payload filters for Qdrant, and where clauses for Chroma. This module abstracts those differences behind a unified filter specification, so the same logical filter (such as "topic equals AI") is automatically translated to the correct database-specific syntax.

## Overview

- Indexes documents with full JSON metadata preservation across all five databases
- Translates a common filter operator syntax into database-native query formats
- Supports eight standard filter operators: equality, inequality, greater than, greater than or equal, less than, less than or equal, inclusion, and exclusion
- Flattens complex metadata structures into primitive types suitable for vector database storage
- Normalizes search results into a consistent format regardless of the underlying database
- All pipelines are configuration-driven through YAML files with environment variable substitution

## How It Works

### Indexing

Each database has a dedicated indexer that loads documents from a configured dataset, generates dense embeddings, flattens document metadata into primitive types, and writes the embedded documents with their metadata to the vector database. The metadata flattening step converts non-primitive types to string representations and skips null values, ensuring compatibility with all database backends.

### Filtered Search

The search pipeline embeds the query, performs a vector similarity search, and applies metadata filters to narrow results. Filters use a common operator syntax that gets translated to each database's native format:

- **Pinecone**: Dictionary-based metadata filters
- **Weaviate**: GraphQL where clauses with operand structure
- **Milvus**: Boolean expression strings (e.g., `field == "value"`)
- **Qdrant**: Payload filter conditions with must/must_not clauses
- **Chroma**: Where clauses with operator mappings

### Filter Specification

The module defines a set of supported operators that work consistently across all databases. These include equality, inequality, numeric comparisons, and set membership checks. Each database-specific filter translator converts from this common format to the native representation.

### Result Normalization

Search results from all databases are normalized to a standard format containing document ID, relevance score, content text, and metadata dictionary. This ensures that downstream consumers receive consistent output regardless of which database was queried.

## Supported Databases

| Database | Status | Filter Translation |
|----------|--------|-------------------|
| Milvus | Supported | Boolean expression strings |
| Pinecone | Supported | Dictionary metadata filters |
| Qdrant | Supported | Payload filter conditions |
| Weaviate | Supported | GraphQL where operands |
| Chroma | Supported | Where clause mappings |

## Configuration

Configuration is stored in YAML files organized by database and dataset under the `configs/` directory. Below is an example:

```yaml
milvus:
  uri: "${MILVUS_URI:-http://localhost:19530}"
  token: "${MILVUS_TOKEN:-}"

collection:
  name: "triviaqa_json_indexed"
  description: "TriviaQA with JSON metadata"

dataloader:
  type: "triviaqa"
  dataset_name: "trivia_qa"
  config: "rc"
  split: "test"
  limit: null

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  batch_size: 32

search:
  top_k: 10

logging:
  level: "INFO"
  name: "milvus_triviaqa_json"
```

## Directory Structure

```
json_indexing/
├── __init__.py                  # Package exports (indexers and searchers)
├── README.md                    # This file
├── common/                      # Shared utilities
│   ├── __init__.py
│   ├── config.py                # YAML config loading with env var resolution
│   ├── embeddings.py            # Embedding model wrappers
│   ├── metadata.py              # Metadata flattening utilities
│   ├── results.py               # Search result normalization
│   └── filters/                 # Database-specific filter translators
│       ├── __init__.py
│       ├── spec.py              # Supported filter operators definition
│       ├── chroma.py            # Chroma where clause translation
│       ├── milvus.py            # Milvus boolean expression translation
│       ├── pinecone.py          # Pinecone dictionary filter translation
│       ├── qdrant.py            # Qdrant payload filter translation
│       └── weaviate.py          # Weaviate GraphQL filter translation
├── indexing/                    # Per-database indexing pipelines
│   ├── __init__.py
│   ├── chroma.py
│   ├── milvus.py
│   ├── pinecone.py
│   ├── qdrant.py
│   └── weaviate.py
├── search/                      # Per-database search pipelines
│   ├── __init__.py
│   ├── chroma.py
│   ├── milvus.py
│   ├── pinecone.py
│   ├── qdrant.py
│   └── weaviate.py
└── configs/                     # YAML configuration files
    ├── chroma/                  # Chroma configs per dataset
    ├── milvus/                  # Milvus configs per dataset
    ├── pinecone/                # Pinecone configs per dataset
    ├── qdrant/                  # Qdrant configs per dataset
    └── weaviate/                # Weaviate configs per dataset
```

## Related Modules

- `src/vectordb/haystack/dense_indexing/` - Dense embedding indexing without metadata filtering
- `src/vectordb/haystack/hybrid_indexing/` - Hybrid sparse-dense indexing and search
- `src/vectordb/utils/converters.py` - Document format converters for each database
