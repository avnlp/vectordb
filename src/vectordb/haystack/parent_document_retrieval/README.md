# Parent Document Retrieval

Hierarchical chunking pipelines that split documents into parent and child chunks to balance retrieval precision with context richness. Only the smaller child chunks are embedded and stored in the vector database, where their compact size enables precise semantic matching against queries. When a child chunk matches, the system returns the corresponding parent chunk instead, providing the downstream language model with a broader context window that captures surrounding information the child alone would miss.

This approach addresses a fundamental tension in retrieval-augmented generation: small chunks match queries more precisely, but large chunks provide more complete context for answer generation. By decoupling the unit of retrieval (child) from the unit of context (parent), the pipeline achieves both goals simultaneously.

## Overview

- Hierarchical document splitting into parent chunks and smaller child chunks
- Child chunks are embedded and indexed for precise semantic matching
- Parent chunks are stored separately and returned when their children match a query
- Auto-merging retrieval automatically resolves child matches to parent documents
- Configurable chunk sizes, overlap, and merge thresholds
- In-memory parent document store for fast parent lookups during search

## How It Works

### Indexing

The indexing pipeline loads documents from a configured dataset and splits them using a hierarchical document splitter that produces two levels: larger parent chunks and smaller child chunks nested within them. Parent-child relationships are tracked via metadata. The parent chunks are written to an in-memory document store for later retrieval. The child (leaf) chunks are embedded using a sentence transformer model and upserted into the vector database. Only the leaf-level chunks receive embeddings, keeping the index focused on fine-grained semantic units.

### Search

The search pipeline embeds the incoming query and searches the vector database for matching child chunks, over-sampling by a factor of three to ensure broad coverage. The matched child chunks are passed to an auto-merging retriever, which looks up the corresponding parent documents from the in-memory store. When enough children from the same parent match the query (controlled by a merge threshold), the parent document is returned in place of its children. The final result set contains parent-level documents that provide richer context than the individual child chunks that triggered the match.

## Supported Databases

| Database | Indexing Module | Search Module | Notes |
|----------|----------------|---------------|-------|
| Pinecone | `indexing/pinecone.py` | `search/pinecone.py` | Serverless managed service |
| Weaviate | `indexing/weaviate.py` | `search/weaviate.py` | GraphQL-based queries |
| Chroma | `indexing/chroma.py` | `search/chroma.py` | Local or cloud deployment |
| Milvus | `indexing/milvus.py` | `search/milvus.py` | Distributed vector database |
| Qdrant | `indexing/qdrant.py` | `search/qdrant.py` | Payload filtering support |

## Configuration

Each database-dataset combination has a dedicated YAML configuration file. The configuration controls chunking parameters (parent and child sizes, overlap), retrieval settings (top-k, merge threshold), embedding model selection, and database connection details.

```yaml
logging:
  name: haystack_parent_doc_qdrant_triviaqa
  level: INFO

embeddings:
  model: Qwen/Qwen3-Embedding-0.6B

chunking:
  child_chunk_size_words: 25
  parent_chunk_size_words: 100
  split_overlap: 5

retrieval:
  top_k: 5
  merge_threshold: 0.5

dataloader:
  type: triviaqa
  dataset_name: trivia_qa
  split: test
  index_limit: 500
  eval_limit: 100

database:
  type: qdrant
  qdrant:
    url: ${QDRANT_URL}
    api_key: ${QDRANT_API_KEY}
    collection_name: parent_doc_children_triviaqa
```

## Directory Structure

```
src/vectordb/haystack/parent_document_retrieval/
├── __init__.py                        # Public exports for all pipeline classes
├── configs/                           # YAML configs (25 files: 5 databases x 5 datasets)
│   ├── qdrant_triviaqa_config.yaml
│   ├── qdrant_arc_config.yaml
│   ├── qdrant_popqa_config.yaml
│   ├── qdrant_factscore_config.yaml
│   ├── qdrant_earnings_calls_config.yaml
│   ├── pinecone_triviaqa_config.yaml
│   ├── ...
│   └── weaviate_earnings_calls_config.yaml
├── indexing/                          # Indexing pipelines with hierarchical splitting
│   ├── __init__.py
│   ├── chroma.py                      # Chroma parent document indexing
│   ├── milvus.py                      # Milvus parent document indexing
│   ├── pinecone.py                    # Pinecone parent document indexing
│   ├── qdrant.py                      # Qdrant parent document indexing
│   └── weaviate.py                    # Weaviate parent document indexing
├── search/                            # Search pipelines with auto-merging retrieval
│   ├── __init__.py
│   ├── chroma.py                      # Chroma parent document search
│   ├── milvus.py                      # Milvus parent document search
│   ├── pinecone.py                    # Pinecone parent document search
│   ├── qdrant.py                      # Qdrant parent document search
│   └── weaviate.py                    # Weaviate parent document search
├── utils/                             # Shared utilities
│   ├── __init__.py
│   ├── config.py                      # Configuration loading with env var resolution
│   ├── hierarchy.py                   # Document hierarchy utilities
│   ├── ids.py                         # Document ID generation
│   └── metadata.py                    # Metadata management for parent-child links
└── README.md
```

## Related Modules

- `src/vectordb/haystack/semantic_search/` - Standard flat dense indexing without hierarchy
- `src/vectordb/haystack/contextual_compression/` - Post-retrieval compression as alternative context strategy
- `src/vectordb/haystack/reranking/` - Two-stage retrieval with cross-encoder reranking
- `src/vectordb/dataloaders/` - Dataset loading for all supported datasets
