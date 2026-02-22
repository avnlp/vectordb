# Namespaces

This module provides namespace isolation for LangChain vector database pipelines across all five supported databases. It enables CRUD operations on namespaces, indexing documents into separate namespaces, querying within a specific namespace, and cross-namespace search with timing comparisons.

Each database uses its native mechanism for data isolation: Pinecone provides native namespace parameters, Weaviate uses tenant-based isolation, Milvus uses partition key field filtering, Qdrant uses payload-based filtering on a shared collection, and Chroma uses separate collections per namespace. The module abstracts these differences behind a unified `NamespacePipeline` interface with consistent typed results, timing metrics, and structured error handling.

## Overview

- Creates, lists, checks existence of, and deletes namespaces using each database's native isolation mechanism
- Indexes documents with pre-computed embeddings into specific namespaces
- Supports namespace-scoped semantic search with optional metadata filters and RAG answer generation
- Supports cross-namespace search to compare query results and timing across multiple namespaces
- Provides namespace statistics including document and vector counts
- Defines typed result objects, custom exceptions, and timing metric collection
- Includes namespace name generators for consistent naming from dataset splits, ticker symbols, or configuration
- Includes query samplers for extracting test queries from document metadata
- Separate indexing and search pipeline classes per database for independent scaling
- All behavior is driven by YAML configuration files with environment variable substitution
- 25 pre-built configuration files covering all database and dataset combinations

## How It Works

### Architecture

The module is organized into three layers:

1. **Core namespace pipelines** (`pinecone.py`, `weaviate.py`, `chroma.py`, `milvus.py`, `qdrant.py`) -- Each extends the `NamespacePipeline` abstract base class and implements the seven abstract namespace operations using the database's native isolation mechanism. These handle low-level namespace CRUD and document upsert with namespace routing. Cross-namespace querying is provided by the base class.

2. **Indexing pipelines** (`indexing/`) -- Each wraps a core namespace pipeline and adds end-to-end document loading, embedding generation, and index creation. Initialized with a YAML config and a target namespace, the `run()` method loads documents via `DataloaderCatalog`, embeds them with `EmbedderHelper`, and indexes them into the specified namespace.

3. **Search pipelines** (`search/`) -- Each wraps a core namespace pipeline and adds query embedding and optional RAG answer generation. The `search()` method embeds the query with `EmbedderHelper`, executes a namespace-scoped similarity search, and optionally generates an answer using `RAGHelper`.

### Base Interface

All database implementations extend `NamespacePipeline`, which defines the following abstract methods:

- `create_namespace(namespace)` -- Creates a new namespace, returning a `NamespaceOperationResult`
- `delete_namespace(namespace)` -- Deletes a namespace and all its data
- `list_namespaces()` -- Returns all namespace identifiers as a list of strings
- `namespace_exists(namespace)` -- Checks whether a namespace exists
- `get_namespace_stats(namespace)` -- Returns a `NamespaceStats` object with document count, vector count, and status
- `index_documents(documents, embeddings, namespace)` -- Indexes LangChain documents with pre-computed embeddings into a namespace
- `query_namespace(query, namespace, top_k)` -- Queries a single namespace, returning a list of `NamespaceQueryResult`

The base class also provides a concrete method:

- `query_cross_namespace(query, namespaces, top_k)` -- Queries multiple namespaces and returns a `CrossNamespaceResult` with per-namespace timing comparisons. This is implemented in the base class using `list_namespaces()` and `query_namespace()`, so subclasses inherit it automatically.

### Namespace Creation and Management

Each database pipeline provides methods to create, list, check existence of, and delete namespaces. The creation behavior differs by database:

- **Pinecone** auto-creates namespaces on first upsert; `create_namespace` is a no-op that returns a success result
- **Weaviate** explicitly creates tenants via `create_tenant()` on the database client
- **Chroma** creates a prefixed collection (e.g., `ns_arc_train`) for each namespace; `list_namespaces` strips the prefix when returning names
- **Milvus** auto-creates partition key namespaces on insert; `list_namespaces` paginates through all entities to collect unique namespace values; `delete_namespace` removes all documents matching the namespace filter expression
- **Qdrant** auto-creates payload-based namespaces on insert; `list_namespaces` scrolls through all points to collect unique namespace payload values

### Indexing Pipelines

Each indexing pipeline (`PineconeNamespaceIndexingPipeline`, `WeaviateNamespaceIndexingPipeline`, etc.) follows a consistent flow:

1. Validates that the namespace is non-empty
2. Loads and validates the YAML configuration using `ConfigLoader`
3. Creates the embedding model using `EmbedderHelper.create_embedder()`
4. Initializes the database-specific `NamespacePipeline`
5. Loads documents from the configured dataset via `DataloaderCatalog`
6. Generates embeddings for all documents with `EmbedderHelper.embed_documents()`
7. For Pinecone, creates the index with configurable metric and recreate options
8. Indexes documents into the target namespace via `pipeline.index_documents()`

The `run()` method returns a dictionary with `documents_indexed` (int) and `namespace` (str).

### Search Pipelines

Each search pipeline (`PineconeNamespaceSearchPipeline`, `WeaviateNamespaceSearchPipeline`, etc.) follows a consistent flow:

1. Validates that the namespace is non-empty
2. Loads and validates the YAML configuration
3. Creates the embedding model and optionally an LLM for RAG via `RAGHelper.create_llm()`
4. Initializes the database-specific `NamespacePipeline`

The `search()` method accepts a query string, `top_k`, and optional metadata `filters`. It:

1. Embeds the query using `EmbedderHelper.embed_query()`
2. Executes a namespace-scoped similarity search on the database
3. If an LLM is configured, generates a RAG answer from the retrieved documents
4. Returns a dictionary with `documents`, `query`, `namespace`, and optionally `answer`

Each database routes the namespace scope differently during search: Pinecone passes `namespace=`, Weaviate passes `tenant=`, Chroma resolves the prefixed `collection_name`, Milvus passes `partition_name=`, and Qdrant passes `namespace=` as a payload filter.

### Cross-Namespace Search

The `query_cross_namespace` method on the base `NamespacePipeline` class runs the same query against multiple namespaces and collects per-namespace timing metrics. If no namespaces are specified, it queries all available namespaces. Results are returned as a `CrossNamespaceResult` containing:

- `namespace_results` -- a dict mapping each namespace to its list of `NamespaceQueryResult` objects
- `timing_comparison` -- a list of `CrossNamespaceComparison` objects with per-namespace timing, result count, and top relevance score
- `total_time_ms` -- total wall time across all namespace queries, measured using the `Timer` context manager

### Types

The `types` module defines the following:

**Enums:**
- `IsolationStrategy` -- `NAMESPACE`, `TENANT`, `PARTITION_KEY`, `PAYLOAD_FILTER`, `COLLECTION`
- `TenantStatus` -- `ACTIVE`, `INACTIVE`, `OFFLOADED`, `UNKNOWN`

**Dataclasses:**
- `NamespaceConfig` -- Configuration for a namespace with name, description, split, and metadata
- `NamespaceStats` -- Statistics with document count, vector count, status, timestamps, and size
- `NamespaceTimingMetrics` -- Timing breakdown with namespace lookup, vector search, total time, and document counts
- `NamespaceQueryResult` -- A retrieved document with relevance score, rank, namespace, and optional timing
- `CrossNamespaceComparison` -- Per-namespace timing summary with result count and top score
- `CrossNamespaceResult` -- Full cross-namespace result with per-namespace results and timing comparison
- `NamespaceOperationResult` -- CRUD operation result with success flag, namespace, operation type, message, and optional data

**Exceptions:**
- `NamespaceError` -- Base exception for namespace operations
- `NamespaceNotFoundError` -- Namespace does not exist
- `NamespaceExistsError` -- Namespace already exists on explicit create
- `NamespaceOperationNotSupportedError` -- Operation not supported by database
- `NamespaceConnectionError` -- Database connection failure

**Utilities:**
- `NamespaceNameGenerator` -- Static methods for generating namespace names from dataset splits (`from_split`), ticker symbols (`from_ticker`), or YAML config (`from_config`)
- `QuerySampler` -- Samples test queries from document metadata with configurable field name and optional random seed

### Utilities

The `utils/` sub-package provides three helpers:

- `config.py` -- `load_config()` reads a YAML file and recursively resolves `${ENV_VAR}` patterns using `os.environ`. `resolve_env_vars()` can be called independently on any config dictionary.
- `data.py` -- `load_documents_from_config()` loads LangChain documents from a dataset using `DataloaderCatalog`, with optional split and limit overrides. `get_namespace_configs()` extracts namespace definitions from a configuration dictionary.
- `timing.py` -- `Timer` is a context manager that measures elapsed wall time in milliseconds using `time.perf_counter()`. Used by cross-namespace queries to capture total query time.

## Supported Databases

| Database | Isolation Mechanism | Strategy Enum | Creation Behavior | Notes |
|----------|-------------------|---------------|-------------------|-------|
| Pinecone | Native namespaces | `NAMESPACE` | Auto-created on first upsert | Index shared across namespaces |
| Weaviate | Multi-tenancy (tenants) | `TENANT` | Explicit tenant creation | Auto-creates on index if missing |
| Chroma | Collection per namespace | `COLLECTION` | Creates prefixed collection | `ns_` prefix by default |
| Milvus | Partition key field | `PARTITION_KEY` | Auto-created on insert | Uses metadata field filtering |
| Qdrant | Payload-based filtering | `PAYLOAD_FILTER` | Auto-created on insert | Scrolls all points to list namespaces |

## Configuration

Configuration is stored in YAML files under the `configs/` directory, with one file per database-dataset combination. Below is an example:

```yaml
dataloader:
  type: "triviaqa"
  split: "test"
  limit: 100
  use_text_splitter: false

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "lc-namespaces-triviaqa"
  dimension: 384
  metric: "cosine"
  recreate: false

search:
  top_k: 10

logging:
  name: "lc_namespaces_pinecone"
  level: "INFO"
```

All indexing and search pipelines accept either a path to a YAML file or a pre-loaded configuration dictionary via the `config_or_path` parameter. Environment variables referenced as `${VAR}` are resolved at load time.

## Directory Structure

```
namespaces/
├── __init__.py                        # Package exports for pipelines, types, and utilities
├── README.md                          # This file
├── base.py                            # Abstract NamespacePipeline base class (7 abstract methods + concrete query_cross_namespace)
├── types.py                           # Enums, dataclasses, exceptions, and utility classes
├── pinecone.py                        # Pinecone native namespace pipeline
├── weaviate.py                        # Weaviate tenant-based namespace pipeline
├── chroma.py                          # Chroma collection-based namespace pipeline
├── milvus.py                          # Milvus partition key namespace pipeline
├── qdrant.py                          # Qdrant payload-filtered namespace pipeline
├── indexing/                          # Namespace indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # PineconeNamespaceIndexingPipeline
│   ├── weaviate.py                    # WeaviateNamespaceIndexingPipeline
│   ├── chroma.py                      # ChromaNamespaceIndexingPipeline
│   ├── milvus.py                      # MilvusNamespaceIndexingPipeline
│   └── qdrant.py                      # QdrantNamespaceIndexingPipeline
├── search/                            # Namespace search pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # PineconeNamespaceSearchPipeline
│   ├── weaviate.py                    # WeaviateNamespaceSearchPipeline
│   ├── chroma.py                      # ChromaNamespaceSearchPipeline
│   ├── milvus.py                      # MilvusNamespaceSearchPipeline
│   └── qdrant.py                      # QdrantNamespaceSearchPipeline
├── utils/                             # Shared utilities
│   ├── __init__.py
│   ├── config.py                      # YAML loading with ${ENV_VAR} resolution
│   ├── data.py                        # Document loading and namespace config extraction
│   └── timing.py                      # Timer context manager for operation timing
└── configs/                           # 25 YAML configs (5 databases x 5 datasets)
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── pinecone_popqa.yaml
    ├── pinecone_factscore.yaml
    ├── pinecone_earnings_calls.yaml
    ├── weaviate_triviaqa.yaml
    ├── weaviate_arc.yaml
    ├── ...
    ├── chroma_*.yaml
    ├── milvus_*.yaml
    └── qdrant_*.yaml
```

## Related Modules

- `src/vectordb/langchain/semantic_search/` - Dense search without namespace isolation
- `src/vectordb/langchain/utils/` - Shared LangChain utilities (`ConfigLoader`, `EmbedderHelper`, `RAGHelper`) used by indexing and search pipelines
- `src/vectordb/haystack/namespaces/` - Haystack equivalent with pipeline-based namespace management
- `src/vectordb/dataloaders/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and EarningsCalls
