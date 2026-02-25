# Parent Document Retrieval

Hierarchical document retrieval that indexes small chunks for precise matching but returns larger parent documents for context. This approach solves the chunk-size tradeoff in retrieval systems: small chunks match queries accurately but lack surrounding context needed for comprehensive answers, while large chunks provide context but match imprecisely.

The module maintains a mapping between chunks and their parent documents, enabling retrieval at the chunk level followed by reconstruction of the full parent context. This is particularly valuable for long documents where different sections might address different aspects of a query.

## Overview

- Indexes small chunks for precise semantic matching
- Returns full parent documents to provide comprehensive context
- Maintains chunk-to-parent mapping in a dedicated store
- Configurable chunk and parent sizes with overlap control
- Multiple retrieval modes: chunks only, parents only, or combined
- Supports all five vector databases with consistent interface
- Parent store persistence for production deployments
- Configuration-driven through YAML files with environment variable substitution

## How It Works

### Indexing Phase

The indexing pipeline processes documents through a two-stage approach. First, parent documents are loaded from the dataset and stored in a parent document store. Each parent document receives a unique identifier. Then, the pipeline splits parent documents into smaller chunks using configurable size and overlap parameters. Each chunk is embedded and indexed into the vector database with metadata linking it to its parent document ID.

The chunking strategy balances precision and coverage. Smaller chunks (256-512 tokens) enable fine-grained matching to specific query terms. Overlap between chunks ensures that content spanning chunk boundaries is not lost. The parent store maintains the complete parent documents indexed by their identifiers.

### Search Phase

The search pipeline embeds the query and retrieves the most similar chunks from the vector database. The chunk metadata includes parent document identifiers, which are extracted from the retrieved chunks. The pipeline then looks up the corresponding parent documents from the parent store, returning the full parent content rather than the small chunks.

This approach ensures that while matching occurs at a fine-grained level (chunks), the returned context is comprehensive (full parents). If multiple chunks from the same parent are retrieved, the parent is returned only once to avoid duplication.

### Parent Store

The parent document store maintains the mapping between chunk IDs and parent documents. It provides:

- **Storage**: In-memory storage of parent documents indexed by ID
- **Persistence**: Optional save/load to disk for production deployments
- **Lookup**: Efficient retrieval of parents by chunk ID or parent ID
- **Deduplication**: Ensures each parent is returned only once even if multiple chunks reference it

The store can be persisted to disk and reloaded across sessions, enabling stateful production deployments where the parent store survives restarts.

### Retrieval Modes

The pipeline supports multiple retrieval modes depending on the use case:

**Parent Only Mode** retrieves chunks, maps to parents, and returns only the unique parent documents. This is the default mode and provides the best balance of precision and context.

**With Context Mode** returns both the matched chunks and their parent documents, allowing the consumer to see exactly which parts matched while still having full context available.

**Context Window Mode** retrieves a parent document and extracts a window around the matched chunk, returning just that portion rather than the full parent.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Pinecone | Supported | Namespace-scoped parent storage |
| Weaviate | Supported | Collection-based parent storage |
| Chroma | Supported | Works with persistent storage |
| Milvus | Supported | Partition-aware parent lookup |
| Qdrant | Supported | Payload-based parent references |

## Configuration

Configuration is stored in YAML files organized by database and dataset. The configuration controls chunking parameters, parent store settings, and retrieval mode.

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "parent-doc-index"
  namespace: ""

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32

chunking:
  chunk_size: 512
  chunk_overlap: 50
  parent_size: 2048  # Size of parent documents

parent_store:
  store_path: "./parent_store.json"  # Optional persistence
  auto_save: true

retrieval:
  mode: "parent_only"  # or "with_context", "context_window"
  top_k: 5
  max_parents: 10

rag:
  enabled: false
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048

logging:
  level: "INFO"
```

## Directory Structure

```
parent_document_retrieval/
├── __init__.py                        # Package exports
├── parent_store.py                    # Parent document storage and mapping
├── indexing/                          # Database-specific indexing pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone parent doc indexing
│   ├── weaviate.py                    # Weaviate parent doc indexing
│   ├── chroma.py                      # Chroma parent doc indexing
│   ├── milvus.py                      # Milvus parent doc indexing
│   └── qdrant.py                      # Qdrant parent doc indexing
├── search/                            # Database-specific search pipelines
│   ├── __init__.py
│   ├── pinecone.py                    # Pinecone parent doc search
│   ├── weaviate.py                    # Weaviate parent doc search
│   ├── chroma.py                      # Chroma parent doc search
│   ├── milvus.py                      # Milvus parent doc search
│   └── qdrant.py                      # Qdrant parent doc search
└── configs/                           # YAML configs organized by database
    ├── pinecone_triviaqa.yaml
    ├── pinecone_arc.yaml
    ├── weaviate_triviaqa.yaml
    └── ...                            # (25+ config files total)
```

## Related Modules

- `src/vectordb/langchain/semantic_search/` - Standard chunk-level semantic search
- `src/vectordb/langchain/contextual_compression/` - Post-retrieval compression of parent docs
- `src/vectordb/langchain/hybrid_indexing/` - Hybrid search with parent document retrieval
- `src/vectordb/dataloaders/` - Dataset loaders with parent document support
