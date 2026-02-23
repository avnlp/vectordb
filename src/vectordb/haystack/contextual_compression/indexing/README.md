# Contextual Compression Indexing Pipelines

Standalone indexing pipelines for contextual compression across all major vector databases.

## Overview

This module provides database-specific indexing pipelines that handle:

1. **Data Loading**: Load documents from HuggingFace datasets (TriviaQA, ARC, PopQA, FactScore, Earnings Calls)
2. **Embedding Generation**: Generate embeddings using SentenceTransformers models
3. **Storage**: Store embedded documents in the vector database with metadata

Each pipeline is **completely independent** and does not depend on `json_indexing` module.

## Supported Databases

- **Milvus**: `MilvusIndexingPipeline`
- **Pinecone**: `PineconeIndexingPipeline`
- **Qdrant**: `QdrantIndexingPipeline`
- **Chroma**: `ChromaIndexingPipeline`
- **Weaviate**: `WeaviateIndexingPipeline`

## Supported Datasets

- **TriviaQA** (`triviaqa`): Open-domain QA dataset with search results
- **ARC** (`arc`): AI2 Reading Comprehension Challenge
- **PopQA** (`popqa`): Population-based QA dataset
- **FactScore** (`factscore`): Fact verification dataset
- **Earnings Calls** (`earnings_calls`): Corporate earnings call transcripts

## Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Create Configuration File

Copy `EXAMPLE_CONFIG.yaml` and customize for your database:

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

dataset:
  type: "triviaqa"
  split: "test"
  limit: 1000  # Optional: limit docs

milvus:
  host: "localhost"
  port: 19530
  collection_name: "triviaqa_compression"
  drop_existing: false

logging:
  name: "indexing"
  level: "INFO"
```

### 3. Run Indexing

```python
from vectordb.haystack.contextual_compression.indexing import MilvusIndexingPipeline

pipeline = MilvusIndexingPipeline("path/to/config.yaml")
result = pipeline.run(batch_size=32)

print(f"Indexed {result['indexed_count']} documents")
```

Or from command line:

```bash
python -m vectordb.haystack.contextual_compression.indexing.milvus_indexing config.yaml
```

## Architecture

### Base Class: `BaseIndexingPipeline`

Abstract base class providing:

- Embedder initialization (from config)
- Dataset loading via `DatasetRegistry`
- Batch embedding generation
- Error handling and logging

**Abstract Methods** (implemented by subclasses):

- `_connect()`: Establish database connection
- `_prepare_collection()`: Create/verify collection schema
- `_store_documents(documents)`: Store embedded documents

### Database-Specific Implementations

Each DB has its own file implementing:

1. **Milvus** (`milvus_indexing.py`):
   - Creates collection with schema: `id, content, embedding, metadata`
   - Uses IP (Inner Product) metric with IVF_FLAT index
   - Stores metadata as JSON string

2. **Pinecone** (`pinecone_indexing.py`):
   - Creates/gets index with configurable metric (cosine, euclidean, dotproduct)
   - Uses ServerlessSpec for cloud deployment
   - Stores content in metadata with size limits

3. **Qdrant** (`qdrant_indexing.py`):
   - Creates collection with COSINE distance
   - Stores content and metadata in payload
   - Uses UUIDs for point IDs

4. **Chroma** (`chroma_indexing.py`):
   - Uses PersistentClient with local storage
   - Simple schema: `content, metadata`
   - Automatic embedding storage

5. **Weaviate** (`weaviate_indexing.py`):
   - Creates class with properties: `content, metadata_json`
   - Batch upserts with embeddings
   - Optional cloud authentication

## Configuration Reference

### Common Options

```yaml
embeddings:
  model: str                          # HuggingFace model ID or alias
  dimension: int                      # Embedding dimension (default: 384)

dataset:
  type: str                           # triviaqa, arc, popqa, factscore, earnings_calls
  name: str                           # Optional HuggingFace dataset ID
  split: str                          # train, test, validation, etc. (default: test)
  limit: int | null                   # Max documents to load (null = all)

logging:
  name: str                           # Logger name
  level: str                          # DEBUG, INFO, WARNING, ERROR
```

### Milvus Options

```yaml
milvus:
  host: str                           # Milvus server host (default: localhost)
  port: int                           # Milvus server port (default: 19530)
  collection_name: str                # Collection name
  drop_existing: bool                 # Drop and recreate collection
```

### Pinecone Options

```yaml
pinecone:
  api_key: str                        # API key (env var: $PINECONE_API_KEY)
  index_name: str                     # Index name
  metric: str                         # cosine, euclidean, dotproduct
  cloud: str                          # aws, gcp, azure
  region: str                         # Cloud region
```

### Qdrant Options

```yaml
qdrant:
  url: str                            # Qdrant server URL
  api_key: str | null                 # API key (optional)
  collection_name: str                # Collection name
```

### Chroma Options

```yaml
chroma:
  path: str                           # Local storage path
  persist_directory: str              # Persist directory
  collection_name: str                # Collection name
```

### Weaviate Options

```yaml
weaviate:
  url: str                            # Weaviate URL
  api_key: str | null                 # API key (optional, for cloud)
  collection_name: str                # Collection name (PascalCase)
```

## Usage Examples

### Milvus

```python
from vectordb.haystack.contextual_compression.indexing import MilvusIndexingPipeline

pipeline = MilvusIndexingPipeline("configs/milvus/triviaqa.yaml")
result = pipeline.run(batch_size=64)

assert result["status"] == "success"
print(f"Indexed {result['indexed_count']} documents")
```

### Pinecone

```python
from vectordb.haystack.contextual_compression.indexing import PineconeIndexingPipeline

# Requires PINECONE_API_KEY env variable
pipeline = PineconeIndexingPipeline("configs/pinecone/arc.yaml")
result = pipeline.run()

if result["status"] == "error":
    print(f"Error: {result['error']}")
```

### Qdrant

```python
from vectordb.haystack.contextual_compression.indexing import QdrantIndexingPipeline

pipeline = QdrantIndexingPipeline("configs/qdrant/popqa.yaml")
result = pipeline.run(batch_size=32)

print(f"Indexed {result['indexed_count']} documents in {result['status']}")
```

### Chroma

```python
from vectordb.haystack.contextual_compression.indexing import ChromaIndexingPipeline

pipeline = ChromaIndexingPipeline("configs/chroma/factscore.yaml")
result = pipeline.run()

if result["status"] == "success":
    print(f"Stored {result['indexed_count']} documents locally")
```

### Weaviate

```python
from vectordb.haystack.contextual_compression.indexing import WeaviateIndexingPipeline

pipeline = WeaviateIndexingPipeline("configs/weaviate/earnings_calls.yaml")
result = pipeline.run(batch_size=16)

print(f"Indexed: {result['indexed_count']}, Status: {result['status']}")
```

## Schema Information

### Milvus Schema

```
Fields:
  - id (INT64, auto_id, primary)
  - content (VARCHAR, max 65535)
  - embedding (FLOAT_VECTOR, dimension configurable)
  - metadata (VARCHAR, JSON string)

Index:
  - IVF_FLAT on embedding field
  - Metric: IP (Inner Product)
  - nlist: 128
```

### Pinecone Schema

```
Vector: embedding
Metadata:
  - content (text, truncated to 50K chars)
  - metadata_json (JSON string)

Metric: configurable (cosine, euclidean, dotproduct)
```

### Qdrant Schema

```
Payload:
  - content (text)
  - metadata_json (JSON string)

Distance: COSINE
```

### Chroma Schema

```
Document: content (text)
Metadata: flat dict (supports nested via serialization)

Automatic vector storage
```

### Weaviate Schema

```
Class: Compression (PascalCase)
Properties:
  - content (TEXT)
  - metadata_json (TEXT)

Vector: embedding (stored separately)
```

## Testing

### Unit Tests

```bash
# Run all unit tests (excludes integration tests)
uv run pytest tests/test_contextual_compression_indexing.py -v -k "not integration"

# Run specific database tests
uv run pytest tests/test_contextual_compression_indexing.py::TestMilvusIndexingPipeline -v

# Run with coverage
uv run pytest tests/test_contextual_compression_indexing.py --cov=src/vectordb/haystack/contextual_compression/indexing
```

### Integration Tests

Integration tests require actual database connections:

```bash
# Test Milvus (requires MILVUS_HOST env var)
export MILVUS_HOST=localhost
uv run pytest tests/test_contextual_compression_indexing.py::test_milvus_indexing_integration -v

# Test Pinecone (requires PINECONE_API_KEY env var)
export PINECONE_API_KEY=your-key
uv run pytest tests/test_contextual_compression_indexing.py::test_pinecone_indexing_integration -v
```

## Error Handling

All pipelines return a status dict:

```python
result = pipeline.run()

if result["status"] == "success":
    print(f"Indexed {result['indexed_count']} documents")
elif result["status"] == "empty_dataset":
    print("No documents loaded from dataset")
else:  # error
    print(f"Error: {result['error']}")
```

## Performance Tips

1. **Batch Size**: Adjust `batch_size` in `run()` method
   - Larger batches = faster (but more memory)
   - Default: 32
   - Try: 64-128 for large embeddings

2. **Dataset Limits**: Use `limit` in config for testing
   ```yaml
   dataset:
     limit: 100  # Index only 100 docs for testing
   ```

3. **Model Selection**: Use smaller models for faster embedding
   ```yaml
   embeddings:
     model: "sentence-transformers/all-MiniLM-L6-v2"  # Fast
     # vs
     model: "sentence-transformers/all-mpnet-base-v2"  # Slower but better quality
   ```

4. **Drop Existing**: Set `drop_existing: true` only when needed
   ```yaml
   milvus:
     drop_existing: false  # Reuse existing collection
   ```

## Migration from json_indexing

If you were using `json_indexing` module:

**Before:**
```python
from vectordb.haystack.json_indexing.milvus_json_indexing import MilvusJsonIndexing
pipeline = MilvusJsonIndexing("config.yaml")
```

**After:**
```python
from vectordb.haystack.contextual_compression.indexing import MilvusIndexingPipeline
pipeline = MilvusIndexingPipeline("config.yaml")
```

Key differences:
- Simpler schema (no complex JSON fields)
- Faster indexing (basic vector storage)
- Designed for contextual compression workflows
- Standalone (no dependencies on json_indexing)

## Logging

All pipelines log to a named logger. Configure logging:

```python
import logging

# Set debug level for indexing
logging.getLogger("indexing").setLevel(logging.DEBUG)

# Or in config
logging:
  name: "indexing"
  level: "DEBUG"
```

## Next Steps

After indexing:

1. **Search with Compression**: Use corresponding search pipeline
   ```python
   from vectordb.haystack.contextual_compression.search.milvus_search import MilvusCompressionSearch
   search = MilvusCompressionSearch("config.yaml")
   result = search.search("What is Paris?")
   ```

2. **RAG Generation**: Enable RAG in search pipeline config
   ```yaml
   rag:
     enabled: true
     llm:
       provider: "groq"
       model: "llama-3.3-70b-versatile"
   ```

3. **Evaluation**: Use evaluation metrics
   ```python
   from vectordb.haystack.contextual_compression.evaluation import evaluate
   metrics = evaluate(search_results, ground_truths)
   ```

## Support

For issues or questions, refer to:
- `CONTEXTUAL_COMPRESSION_SIMPLIFICATION_PLAN.md` for architecture
- `tests/test_contextual_compression_indexing.py` for usage patterns
- Individual pipeline files for database-specific details
