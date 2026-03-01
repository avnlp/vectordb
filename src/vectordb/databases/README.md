# Databases

This module contains backend wrappers that give Haystack and LangChain feature pipelines a consistent interface to five vector databases: Chroma, Milvus, Pinecone, Qdrant, and Weaviate.

## What This Layer Does

Each wrapper class handles the low-level details of connecting, creating collections or indexes, inserting and querying documents, and converting between framework document objects (Haystack `Document`, LangChain `Document`) and the native formats each database expects. The feature modules in `haystack/` and `langchain/` depend on these wrappers to stay mostly database-agnostic.

The wrappers expose a consistent set of operations:

- Creating and deleting collections or indexes.
- Inserting (upsert) documents with embeddings and metadata.
- Querying with dense vectors, sparse vectors, or both (hybrid).
- Filtering results by metadata conditions.
- Managing multi-tenancy via tenants, partitions, or namespaces.
- Converting raw database results back to Haystack or LangChain `Document` objects.

## Backend Details

### ChromaVectorDB (`chroma.py`)

Connects to Chroma using one of three client modes: `EphemeralClient` for in-memory use, `PersistentClient` for local file-based storage, or `HttpClient` for remote or Chroma Cloud deployments. The choice is made automatically based on whether a `host` is provided and whether `persistent=True`.

All metadata is automatically flattened before storage because Chroma requires flat key-value pairs with scalar values. Nested dictionaries are converted to dot-notation keys (`"parent.child"`). Lists of uniform scalar values are preserved; mixed or complex lists are serialized to strings.

The `search()` method uses Chroma's experimental `Search` API (available on hosted/cloud Chroma 0.6+) when a `host` is configured. For local clients it always falls back to the standard `query()` method. Distance scores (0 to 2 for cosine) are converted to similarity scores (0 to 1) before being attached to returned documents.

Multi-tenant deployments use the `with_tenant()` method to clone the wrapper with a different tenant and database context.

**Key configuration fields:**
- `host`: Chroma server hostname for remote connections. Omit for local mode.
- `port`: Server port, default 8000.
- `api_key`: API key for Chroma Cloud or authenticated instances.
- `tenant` / `database`: Multi-tenant isolation context.
- `path`: Local storage path for `PersistentClient`, default `"./chroma"`.
- `persistent`: Whether to use persistent storage when no host is set, default `True`.

**Environment variables:** `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`.

---

### MilvusVectorDB (`milvus.py`)

Connects to Milvus (self-hosted) or Zilliz Cloud (managed) using `pymilvus.MilvusClient`. For Zilliz Cloud, both `uri` (the HTTPS endpoint) and `token` (the API key) are required.

Collections are created with a fixed schema: an auto-generated `INT64` primary key, a `FLOAT_VECTOR` dense embedding field, an optional `SPARSE_FLOAT_VECTOR` field for hybrid search, a `VARCHAR` content field, and a `JSON` metadata field. HNSW indexing with cosine similarity is applied to the dense field automatically. If `use_sparse=True`, a `SPARSE_INVERTED_INDEX` with inner product metric is added for the sparse field.

For multi-tenancy, `use_partition_key=True` adds a `VARCHAR` partition key field. Milvus physically routes documents to different partitions based on this key value, making tenant-scoped queries more efficient than post-search filtering.

Filters are expressed as Milvus boolean expression strings. The `build_filter_expression()` method converts Python filter dictionaries (supporting `$eq`, `$gt`, `$lt`, `$in`, `$contains` operators) into these expression strings. Metadata fields are accessed via JSON path notation (`metadata["key"]`).

Hybrid search (`hybrid_search()`) issues two `AnnSearchRequest` objects simultaneously (one dense, one sparse) and merges results using `RRFRanker` (Reciprocal Rank Fusion) by default, or `WeightedRanker` when explicit weight control is needed.

**Key configuration fields:**
- `uri`: Milvus server URI. Default `"http://localhost:19530"`.
- `token`: API token for Zilliz Cloud. Empty string for local Milvus.
- `collection_name`: Default collection for operations.

---

### PineconeVectorDB (`pinecone.py`)

Connects to Pinecone using the GRPC client (`PineconeGRPC`) for lower latency. The client and index handle are lazy-initialized on first use. This allows the wrapper to be configured from YAML or environment variables before any network connection is established.

Indexes are created as serverless (AWS `us-east-1` by default) with `cosine`, `euclidean`, or `dotproduct` metrics. After creation, `wait_for_index_ready()` polls every 5 seconds until the index reports `ready=True`.

Pinecone's namespace feature maps directly to logical data partitions. All upsert and query operations accept a `namespace` parameter. The `delete_namespace()` method removes all vectors from a namespace. Namespaces are retrieved from index statistics using `list_namespaces()`.

Metadata must be flat: nested dictionaries are flattened using underscore notation (`"user_id"` from `{"user": {"id": ...}}`). Lists are preserved if all elements are strings; otherwise each element is converted to a string.

Hybrid search (`query_with_sparse()`) passes both `vector` (dense) and `sparse_vector` (in Pinecone `{"indices": [...], "values": [...]}` format) to Pinecone's native hybrid query endpoint.

Filter expressions use Pinecone's MongoDB-style JSON format. `build_filter()` constructs single-field filters and `build_compound_filter()` combines them with `$and` or `$or` logic.

**Key configuration fields:**
- `api_key`: Pinecone API key.
- `index_name`: Name of the Pinecone index.

**Environment variables:** `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.

---

### QdrantVectorDB (`qdrant.py`)

Connects to Qdrant using the `qdrant-client` library with gRPC preferred for production (lower latency). Supports local `http://localhost:6333` or Qdrant Cloud HTTPS endpoints.

Collections can be configured for dense-only or hybrid (named vectors) mode. In hybrid mode, `use_sparse=True` creates separate named vector spaces — `"dense"` and `"sparse"` — within the same Qdrant collection. The vector names are configurable via `dense_vector_name` and `sparse_vector_name` in the config.

Memory-efficient storage is supported via quantization: scalar (INT8) quantization reduces memory by ~4x with minimal accuracy loss; binary quantization achieves ~32x reduction. Quantization is configured using the `quantization` config section with `type: "scalar"` or `type: "binary"`.

Metadata-based tenant isolation uses Qdrant's `is_tenant` payload index optimization. The `_get_tenant_filter()` method constructs a `Filter` object that matches on the `tenant_id` payload field. The `create_namespace_index()` method creates a `keyword` payload index on the partition key field to optimize filter speed.

MMR (Maximal Marginal Relevance) reranking is implemented directly in the wrapper using document embedding vectors. The `mmr_rerank()` method iteratively selects documents that maximize `lambda * relevance - (1 - lambda) * redundancy` using cosine similarity.

Filters use MongoDB-style operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`. The `_build_filter()` method converts these dictionaries to native Qdrant `Filter` objects.

**Key configuration fields:**
- `url`: Qdrant server URL.
- `api_key`: Authentication token for Qdrant Cloud.
- `collection_name`: Target collection.
- `dense_vector_name` / `sparse_vector_name`: Named vector spaces for hybrid mode.
- `quantization.type`: `"scalar"` or `"binary"`.

**Environment variables:** `QDRANT_URL`, `QDRANT_API_KEY`.

---

### WeaviateVectorDB (`weaviate.py`)

Connects to Weaviate Cloud using the v4 client (`weaviate.connect_to_weaviate_cloud`). The connection is established eagerly in `__init__` so configuration errors are caught immediately.

Collections are created with optional vectorizer configurations (for example, `Configure.Vectorizer.text2vec_openai()` to have Weaviate generate embeddings) and generative configurations (for example, `Configure.Generative.openai()` for built-in RAG). Multi-tenancy is enabled per collection with `enable_multi_tenancy=True`.

The `upsert()` method uses Weaviate's batch API (`collection.batch.dynamic()`) for efficient bulk inserts. Vectors and UUIDs are extracted from each record before batch submission. Failed objects are tracked and a `RuntimeError` is raised if any fail.

Querying supports dense vector search (`near_vector`), text-based semantic search (`near_text`), and hybrid search (vector + BM25 via the `hybrid` parameter and `alpha` weight). Generative search (`generate()`) uses Weaviate's built-in RAG with either `single_prompt` (per-document) or `grouped_task` (all documents combined) modes.

The `with_tenant()` method switches the collection context to a specific tenant. All subsequent operations are scoped to that tenant. Tenants can be created and deleted with `create_tenants()` and `delete_tenants()`.

Distance scores (lower = better) are converted to similarity scores (higher = better) using `1 - distance`. Named vectors from Weaviate's vector dictionary are resolved by taking the `"default"` vector or the first available vector.

**Key configuration fields:**
- `cluster_url`: Full URL of the Weaviate cluster.
- `api_key`: Weaviate Cloud API key.
- `headers`: Additional HTTP headers (for example, `{"X-OpenAI-Api-Key": "..."}` for vectorizer access).

## How to Choose a Backend

| Factor | Recommendation |
|---|---|
| Need a fully managed service | Pinecone (serverless) or Zilliz Cloud (Milvus-compatible) |
| Need local development without a server | Chroma (persistent or ephemeral mode) |
| Need built-in BM25 or generative AI integration | Weaviate |
| Need partition-key-based multi-tenancy at scale | Milvus |
| Need strong payload filtering with quantization | Qdrant |

Choose a backend based on your operational model first, then on the retrieval features you need. Keep one baseline backend consistent across all feature experiments before switching.

## Common Mistakes

- Switching backends between experiments without revisiting filter syntax. Each backend uses different filter formats (`$and`/`$eq` for Pinecone, Milvus boolean expressions, Qdrant `Filter` objects, Weaviate `Filter` classes).
- Assuming identical performance characteristics across backends for the same embedding dimension and dataset size.
- Tuning retrieval hyperparameters before validating data ingestion quality and embedding model choice.
- Ignoring metadata normalization requirements: Chroma requires flat metadata, Pinecone requires scalar values or string lists, Milvus uses a JSON field, Qdrant and Weaviate use native payload/property schemas.
