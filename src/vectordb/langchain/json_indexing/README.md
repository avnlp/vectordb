# JSON Indexing Pipelines (LangChain)

This module provides structured JSON document indexing pipelines for LangChain across all five supported vector databases. Each pipeline indexes documents with rich JSON metadata preservation, enabling semantic search over JSON documents while maintaining structured fields for filtering during search.

## Overview

The JSON indexing pipelines follow a consistent three-phase pattern across all database backends:

1. **Document Loading**: Load JSON documents from configured dataloaders (TriviaQA, ARC, Earnings Calls, FActScore, PopQA)
2. **Embedding Generation**: Generate dense embeddings using configurable embedding models
3. **Vector Store Indexing**: Create collections and upsert documents with embeddings and preserved JSON metadata

### Key Features

- **JSON Metadata Preservation**: Full JSON structure maintained in document metadata for filtering
- **Multi-Database Support**: Consistent API across Chroma, Milvus, Pinecone, Qdrant, and Weaviate
- **Configuration-Driven**: YAML-based configuration with environment variable substitution
- **LangChain Integration**: Native integration with LangChain document loaders and utilities
- **Batch Processing**: Efficient batch embedding and upsert operations
- **Empty Set Handling**: Graceful handling of empty document batches

## Supported Databases

| Database | Pipeline Class | Storage Type | Metadata Support |
|----------|---------------|--------------|------------------|
| Chroma | `ChromaJsonIndexingPipeline` | Local/In-memory | Dictionary metadata |
| Milvus | `MilvusJsonIndexingPipeline` | Cloud-native | Dynamic fields |
| Pinecone | `PineconeJsonIndexingPipeline` | Managed cloud | JSON metadata |
| Qdrant | `QdrantJsonIndexingPipeline` | Self-hosted/Cloud | Payload storage |
| Weaviate | `WeaviateJsonIndexingPipeline` | Cloud/Self-hosted | Typed properties |

## Installation

Ensure you have the required dependencies installed:

```bash
pip install vectordb[langchain]
```

Install database-specific dependencies as needed:

```bash
# For Chroma
pip install chromadb

# For Milvus
pip install pymilvus

# For Pinecone
pip install pinecone-client

# For Qdrant
pip install qdrant-client

# For Weaviate
pip install weaviate-client
```

## Quick Start

### Basic Usage

```python
from vectordb.langchain.json_indexing.indexing import ChromaJsonIndexingPipeline

# Load from YAML configuration
pipeline = ChromaJsonIndexingPipeline("configs/chroma/triviaqa.yaml")

# Run indexing
result = pipeline.run()
print(f"Indexed {result['documents_indexed']} documents")
```

### Programmatic Configuration

```python
from vectordb.langchain.json_indexing.indexing import QdrantJsonIndexingPipeline

config = {
    "dataloader": {
        "type": "triviaqa",
        "split": "test",
        "limit": 100
    },
    "embeddings": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32
    },
    "qdrant": {
        "url": "http://localhost:6333",
        "collection_name": "json_docs"
    }
}

pipeline = QdrantJsonIndexingPipeline(config)
result = pipeline.run()
```

## Configuration Reference

### YAML Configuration Structure

```yaml
# Dataloader configuration
dataloader:
  type: "triviaqa"           # Dataset type
  dataset_name: "trivia_qa"  # HuggingFace dataset name
  config: "rc"               # Dataset configuration
  split: "test"              # Dataset split
  limit: null                # Optional document limit

# Embedding model configuration
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Model name
  device: "cpu"               # Device (cpu/cuda)
  batch_size: 32              # Embedding batch size

# Database-specific configuration
chroma:
  path: "./chroma_data"       # Local storage path
  collection_name: "json_docs" # Collection name
  recreate: false             # Recreate collection if exists

# Search configuration (for compatibility with search pipelines)
search:
  top_k: 10                   # Default number of results

# Logging configuration
logging:
  level: "INFO"               # Log level
  name: "chroma_json_indexing" # Logger name
```

### Environment Variables

Configuration files support environment variable substitution using `${VAR}` or `${VAR:-default}` syntax:

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY:-}"
  index_name: "json-docs"

weaviate:
  cluster_url: "${WEAVIATE_CLUSTER_URL:-http://localhost:8080}"
  api_key: "${WEAVIATE_API_KEY:-}"
```

## Pipeline Classes

### ChromaJsonIndexingPipeline

Local persistent storage with collection-based organization. Ideal for development and testing.

```python
from vectordb.langchain.json_indexing.indexing import ChromaJsonIndexingPipeline

config = {
    "chroma": {
        "path": "./data",
        "collection_name": "json_docs"
    },
    "embeddings": {"model": "all-MiniLM-L6-v2"},
    "dataloader": {"type": "triviaqa", "split": "test"}
}

pipeline = ChromaJsonIndexingPipeline(config)
result = pipeline.run()
```

### MilvusJsonIndexingPipeline

Cloud-native vector database with dynamic field support for flexible JSON schemas.

```python
from vectordb.langchain.json_indexing.indexing import MilvusJsonIndexingPipeline

config = {
    "milvus": {
        "uri": "http://localhost:19530",
        "collection_name": "json_docs"
    },
    "embeddings": {"model": "all-MiniLM-L6-v2"},
    "dataloader": {"type": "triviaqa", "split": "test"}
}

pipeline = MilvusJsonIndexingPipeline(config)
result = pipeline.run()
```

### PineconeJsonIndexingPipeline

Fully managed cloud service with namespace support for multi-tenant scenarios.

```python
from vectordb.langchain.json_indexing.indexing import PineconeJsonIndexingPipeline

config = {
    "pinecone": {
        "api_key": "${PINECONE_API_KEY}",
        "index_name": "json-docs",
        "namespace": "default"
    },
    "embeddings": {"model": "all-MiniLM-L6-v2"},
    "dataloader": {"type": "triviaqa", "split": "test"}
}

pipeline = PineconeJsonIndexingPipeline(config)
result = pipeline.run()
```

### QdrantJsonIndexingPipeline

Self-hosted or cloud deployment with advanced payload filtering capabilities.

```python
from vectordb.langchain.json_indexing.indexing import QdrantJsonIndexingPipeline

config = {
    "qdrant": {
        "url": "http://localhost:6333",
        "collection_name": "json_docs"
    },
    "embeddings": {"model": "all-MiniLM-L6-v2"},
    "dataloader": {"type": "triviaqa", "split": "test"}
}

pipeline = QdrantJsonIndexingPipeline(config)
result = pipeline.run()
```

### WeaviateJsonIndexingPipeline

Knowledge graph-enhanced vector database with typed property schemas.

```python
from vectordb.langchain.json_indexing.indexing import WeaviateJsonIndexingPipeline

config = {
    "weaviate": {
        "cluster_url": "https://your-instance.weaviate.network",
        "api_key": "${WEAVIATE_API_KEY}",
        "collection_name": "JsonDocs"
    },
    "embeddings": {"model": "all-MiniLM-L6-v2"},
    "dataloader": {"type": "triviaqa", "split": "test"}
}

pipeline = WeaviateJsonIndexingPipeline(config)
result = pipeline.run()
```

## Supported Datasets

The indexing pipelines support multiple dataset types through the `DataloaderCatalog`:

| Dataset | Type | Description |
|---------|------|-------------|
| TriviaQA | `triviaqa` | Reading comprehension dataset with question-answer pairs |
| ARC | `arc` | AI2 Reasoning Challenge with science exam questions |
| Earnings Calls | `earnings_calls` | Transcripts from corporate earnings calls |
| FActScore | `factscore` | Fact verification dataset |
| PopQA | `popqa` | Popular questions dataset |

## Return Values

All pipelines return a dictionary with indexing statistics:

```python
result = pipeline.run()
# Chroma, Milvus, Qdrant, Weaviate:
# {
#     "documents_indexed": 1000,
#     "collection_name": "json_docs"
# }

# Pinecone:
# {
#     "documents_indexed": 1000,
#     "index_name": "json-docs",
#     "namespace": "default"
# }
```

## Architecture

### Indexing Flow

```
┌─────────────────┐
│  Configuration  │
│  (YAML/Dict)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Dataloader     │
│  (Load JSON)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embedder       │
│  (Generate      │
│  Vectors)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector DB      │
│  (Create        │
│  Collection)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Upsert         │
│  (Documents +   │
│  Embeddings)    │
└─────────────────┘
```

### JSON Handling

1. **Loading**: JSON documents loaded with structure preserved in `document.metadata`
2. **Embedding**: Text content extracted from `document.page_content` for embedding
3. **Storage**: Full JSON structure maintained in database for filtering
4. **Search**: JSON fields accessible via metadata filters in search pipelines

## Directory Structure

```
langchain/json_indexing/
├── __init__.py                  # Package exports
├── indexing/                    # Indexing pipelines (this module)
│   ├── __init__.py
│   ├── chroma.py
│   ├── milvus.py
│   ├── pinecone.py
│   ├── qdrant.py
│   └── weaviate.py
├── search/                      # Search pipelines
│   ├── __init__.py
│   ├── chroma.py
│   ├── milvus.py
│   ├── pinecone.py
│   ├── qdrant.py
│   └── weaviate.py
├── configs/                     # YAML configurations
│   ├── chroma/
│   ├── milvus/
│   ├── pinecone/
│   ├── qdrant/
│   └── weaviate/
└── README.md                    # This documentation
```

## Related Modules

- `src/vectordb/langchain/json_indexing/search/` - Search pipelines for querying indexed documents
- `src/vectordb/langchain/agentic_rag/indexing/` - Agentic RAG indexing pipelines
- `src/vectordb/langchain/hybrid_indexing/indexing/` - Hybrid dense-sparse indexing
- `src/vectordb/haystack/json_indexing/indexing/` - Haystack equivalent implementation

## Testing

Test files are located in `tests/langchain/json_indexing/` and include:

- Indexing pipeline initialization tests
- Document indexing with metadata extraction
- Empty batch handling
- Search pipeline integration
- RAG mode with JSON context

Run tests with:

```bash
pytest tests/langchain/json_indexing/ -v
```

## Best Practices

1. **Use Configuration Files**: Store database connection details in YAML files for easy environment switching
2. **Environment Variables**: Use environment variables for sensitive credentials (API keys, passwords)
3. **Document Limits**: Use `limit` parameter during development to test with smaller datasets
4. **Collection Recreation**: Set `recreate: true` during development, `false` in production
5. **Batch Size**: Tune `batch_size` based on available memory and embedding model requirements
6. **Monitoring**: Enable `INFO` level logging to monitor indexing progress

## Troubleshooting

### Common Issues

**No documents indexed**: Check dataloader configuration and dataset availability. Verify the dataset exists on HuggingFace.

**Connection errors**: Ensure database service is running and connection parameters (URL, port, API key) are correct.

**Dimension mismatch**: Verify embedding model produces consistent dimensions. All documents must have same embedding size.

**Memory errors**: Reduce `batch_size` in embeddings configuration or use document limits during development.

## License

This module is part of the vectordb package. See the main LICENSE file for licensing information.
