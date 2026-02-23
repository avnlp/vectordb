# Cost-Optimized RAG Implementation

**Status**: ✅ Fully Implemented  
**Date**: 2026-01-26

## Overview

This module implements a production-ready, cost-optimized RAG (Retrieval-Augmented Generation) pipeline that supports 5 vector databases and uses native Haystack components.

### Key Features

- ✅ **5 Vector Database Backends**: Milvus, Pinecone, Qdrant, Weaviate, Chroma
- ✅ **Native Haystack Components**: No custom wrappers - uses official components only
- ✅ **Groq LLM Integration**: OpenAI-compatible API for cost-effective generation
- ✅ **Configurable RAG Pipelines**: YAML-based configuration with full control
- ✅ **Production-Ready**: Full test coverage, type hints, linting
- ✅ **Cost-Optimized**: Efficient batching, optional reranking, flexible models

## Architecture

### Component Stack

```
Indexing:
  DatasetRegistry → SentenceTransformersDocumentEmbedder → VectorDB

Search & RAG:
  Query → SentenceTransformersTextEmbedder → VectorDB Retriever
         → (optional) SentenceTransformersSimilarityRanker
         → PromptBuilder → OpenAIGenerator (Groq)
```

### Directory Structure

```
cost_optimized_rag/
├── base/
│   ├── config.py              # Pydantic configuration models
│   ├── chunking.py            # Text chunking utilities
│   ├── fusion.py              # RRF/weighted fusion
│   ├── metrics.py             # Evaluation metrics
│   └── sparse_indexing.py     # Sparse indexing support
├── indexing/
│   ├── milvus_indexer.py      # Milvus indexing pipeline
│   ├── pinecone_indexer.py    # Pinecone indexing pipeline
│   ├── qdrant_indexer.py      # Qdrant indexing pipeline
│   ├── weaviate_indexer.py    # Weaviate indexing pipeline
│   └── chroma_indexer.py      # Chroma indexing pipeline
├── search/
│   ├── milvus_searcher.py     # Milvus search + RAG pipeline
│   ├── pinecone_searcher.py   # Pinecone search + RAG pipeline
│   ├── qdrant_searcher.py     # Qdrant search + RAG pipeline
│   ├── weaviate_searcher.py   # Weaviate search + RAG pipeline
│   └── chroma_searcher.py     # Chroma search + RAG pipeline
├── utils/
│   ├── common.py              # Shared utilities
│   └── prompt_templates.py    # RAG prompt templates
├── evaluation/
│   └── evaluator.py           # Evaluation metrics
├── examples/
│   └── index_and_search.py    # Usage examples
├── configs/                   # Database-specific YAML configs
│   ├── milvus/
│   ├── pinecone/
│   ├── qdrant/
│   ├── weaviate/
│   └── chroma/
└── README.md
```

## Configuration

### Configuration Schema

All pipelines are configured via YAML files. Example structure:

```yaml
# Dataloader settings
dataloader:
  type: triviaqa
  dataset_name: trivia_qa
  split: test
  limit: null

# Embedding settings
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32

# Collection/Index settings
collection:
  name: triviaqa_milvus_rag
  description: TriviaQA Milvus Cost-Optimized RAG

# Vector database-specific settings
milvus:
  host: localhost
  port: 19530

# Indexing settings
indexing:
  vector_config:
    size: 384
    distance: Cosine
  quantization:
    enabled: false

# Search settings
search:
  top_k: 10
  reranking_enabled: false

# Reranker settings (if enabled)
reranker:
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 5

# RAG Generator settings
generator:
  enabled: true
  provider: groq
  model: llama-3.3-70b-versatile
  api_key: ${GROQ_API_KEY}
  api_base_url: https://api.groq.com/openai/v1
  temperature: 0.7
  max_tokens: 1024

# Logging
logging:
  name: milvus_triviaqa_rag
  level: INFO
```

### Environment Variables

Configuration supports environment variable resolution:

```yaml
api_key: ${GROQ_API_KEY}                    # Required variable
timeout: ${REQUEST_TIMEOUT:-30}             # With default value
```

## Usage

### Indexing Documents

```python
from vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer import MilvusIndexingPipeline

# Load config and run indexing
pipeline = MilvusIndexingPipeline("configs/milvus/triviaqa.yaml")
pipeline.run()
```

### Searching

```python
from vectordb.haystack.cost_optimized_rag.search.milvus_searcher import MilvusSearchPipeline

# Initialize pipeline
searcher = MilvusSearchPipeline("configs/milvus/triviaqa.yaml")

# Search only
results = searcher.search("What is the capital of France?", top_k=5)
for result in results:
    print(f"Score: {result['score']}, Content: {result['content']}")
```

### RAG with LLM Generation

```python
# Search with RAG generation
result = searcher.search_with_rag("What is the capital of France?", top_k=5)
print(f"Answer: {result['answer']}")
print(f"Documents: {len(result['documents'])}")
```

## Native Haystack Components

### Document Embedder (Indexing)

```python
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
)
embedder.warm_up()
result = embedder.run(documents=documents)
embedded_docs = result["documents"]
```

### Text Embedder (Search)

```python
from haystack.components.embedders import SentenceTransformersTextEmbedder

embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
embedder.warm_up()
result = embedder.run(text="What is AI?")
embedding = result["embedding"]
```

### Similarity Ranker (Reranking)

```python
from haystack.components.rankers import SentenceTransformersSimilarityRanker

ranker = SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=5
)
ranker.warm_up()
result = ranker.run(query="test", documents=docs)
ranked_docs = result["documents"]
```

### LLM Generator (Groq)

```python
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

generator = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)
```

### Prompt Builder

```python
from haystack.components.builders import PromptBuilder

prompt_template = """
Given these documents, answer the question.

Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer:
"""

builder = PromptBuilder(template=prompt_template)
```

## Utilities

### Common Functions

```python
from vectordb.haystack.cost_optimized_rag.utils.common import (
    load_documents_from_config,
    create_logger,
    format_search_results,
)

# Load documents from dataset registry
documents = load_documents_from_config(config)

# Create logger from config
logger = create_logger(config)

# Format Haystack documents as dicts
results = format_search_results(documents, include_embeddings=True)
```

### Prompt Templates

```python
from vectordb.haystack.cost_optimized_rag.utils.prompt_templates import (
    RAG_ANSWER_TEMPLATE,
    RAG_ANSWER_WITH_SOURCES_TEMPLATE,
    COST_OPTIMIZED_RAG_TEMPLATE,
)
```

## Testing

All components have comprehensive unit and integration tests.

### Unit Tests

```bash
uv run pytest tests/haystack/cost_optimized_rag/ -v
```

### Integration Tests

Requires running vector database instances:

```bash
# Set environment variables
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
export GROQ_API_KEY=your_groq_key

# Run integration tests
uv run pytest tests/haystack/cost_optimized_rag/ -v -m integration_test
```

### Test Files

- `test_milvus_cost_optimized_rag.py` - Milvus pipeline tests
- `test_pinecone_cost_optimized_rag.py` - Pinecone pipeline tests
- `test_qdrant_cost_optimized_rag.py` - Qdrant pipeline tests
- `test_weaviate_cost_optimized_rag.py` - Weaviate pipeline tests
- `test_chroma_cost_optimized_rag.py` - Chroma pipeline tests
- `test_utils_and_config.py` - Utilities and configuration tests

## Code Quality

### Type Checking

```bash
uv run mypy src/vectordb/haystack/cost_optimized_rag
```

### Linting

```bash
uv run ruff check src/vectordb/haystack/cost_optimized_rag --fix
```

### Formatting

```bash
uv run ruff format src/vectordb/haystack/cost_optimized_rag
```

## Performance Considerations

### Cost Optimization

1. **Batch Indexing**: Documents are indexed in configurable batch sizes
2. **Optional Reranking**: Enable only when precision is critical
3. **Model Selection**: Use smaller models for cost reduction (all-MiniLM-L6-v2)
4. **Groq LLM**: Cost-effective alternative to OpenAI (when available)

### Scalability

1. **Milvus**: Scales to billions of vectors with partitioning
2. **Pinecone**: Managed service, built-in scaling
3. **Qdrant**: In-memory + persistent storage options
4. **Weaviate**: Cloud and on-premise options
5. **Chroma**: Lightweight, good for development

## Migration from Legacy Code

### What Was Removed

- ❌ `cost_optimized_rag.py` - Broken syntax, legacy code
- ❌ `base/embeddings.py` - Custom ONNXEmbedder (replaced by native Haystack)
- ❌ `base/reranking.py` - Custom HybridReranker (replaced by native Haystack)
- ❌ `base/pipeline_base.py` - Over-engineered base classes (utility functions only)

### What Was Kept

- ✅ `base/config.py` - Configuration models (with generator settings added)
- ✅ `base/chunking.py` - Text chunking utilities
- ✅ `base/fusion.py` - RRF/weighted fusion for hybrid search
- ✅ `base/metrics.py` - Evaluation metrics

## Supported Datasets

The implementation supports all datasets from `DatasetRegistry`:

- TriviaQA
- ARC (AI2 Reasoning Challenge)
- PopQA
- FactScore
- Earnings Calls

## Next Steps

1. **Production Deployment**: Use your cloud vector database (Pinecone, Weaviate Cloud)
2. **Custom Datasets**: Extend `DatasetRegistry` for proprietary data
3. **Fine-tuning**: Fine-tune embedding models for domain-specific tasks
4. **Evaluation**: Use metrics in `evaluation/evaluator.py` to measure RAG quality

---

**For detailed implementation notes, see**: [notes/features/COST_OPTIMIZED_RAG_SIMPLIFICATION_PLAN.md](../../../../notes/features/COST_OPTIMIZED_RAG_SIMPLIFICATION_PLAN.md)
