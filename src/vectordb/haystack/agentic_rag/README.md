# Agentic RAG

This module implements self-reflecting Retrieval-Augmented Generation pipelines with intelligent query routing. An LLM-based router decides at each step whether to perform a knowledge base retrieval, delegate to web search, invoke a calculation tool, or apply multi-step reasoning. A self-reflection loop then evaluates the quality of the generated answer and may trigger additional retrieval and refinement cycles before producing the final response.

Each supported vector database has a dedicated pipeline implementation that extends a shared base class. The base class handles dataset loading, document chunking and embedding, query routing, answer generation, self-reflection, and evaluation. Database-specific subclasses implement the connection, indexing, and retrieval logic appropriate to each backend.

## Overview

- Routes incoming queries to the most appropriate tool (retrieval, web search, calculation, or reasoning) using LLM-based classification
- Supports iterative self-reflection that evaluates answer quality across relevance, completeness, and grounding dimensions
- Refines answers that fall below a configurable quality threshold, up to a maximum number of iterations
- Provides dataset loading, chunking, embedding, and indexing as part of the pipeline lifecycle
- Includes evaluation capabilities with contextual recall, precision, answer relevancy, and faithfulness metrics
- Offers retry logic with exponential backoff and fallback tool selection for robustness
- All behavior is driven by YAML configuration files with 25 pre-built configs (5 databases times 5 datasets)

## How It Works

### Query Routing

When a query enters the pipeline, the agentic router classifies it into one of four categories. Factual questions about the knowledge base are routed to the retrieval tool. Current events questions are directed to web search (which falls back to retrieval in the current implementation). Mathematical or computational questions are sent to the calculation tool. Complex analytical questions requiring multi-step logic are handled by the reasoning tool.

### Retrieval and Answer Generation

For retrieval-routed queries, the pipeline embeds the query, searches the vector database for the most relevant documents, and passes the retrieved context along with the original query to a language model for answer generation.

### Self-Reflection Loop

When self-reflection is enabled, the pipeline evaluates each generated answer on three dimensions: relevance (does it address the query), completeness (is it sufficiently detailed), and grounding (is it supported by the retrieved context). Each dimension receives a score from zero to one hundred. If the average score falls below the configured quality threshold, the pipeline refines the answer using feedback about the identified weaknesses. This cycle repeats up to the configured maximum number of iterations.

### Evaluation

The pipeline can run evaluation across all loaded questions, collecting metrics on document retrieval counts, refinement rates, tool distribution, and answer quality scores.

## Supported Databases

| Database | Status | Notes |
|----------|--------|-------|
| Qdrant | Supported | Full indexing, routing, and self-reflection |
| Weaviate | Supported | Full indexing, routing, and self-reflection |
| Milvus | Supported | Full indexing, routing, and self-reflection |
| Pinecone | Supported | Full indexing, routing, and self-reflection |
| Chroma | Supported | Full indexing, routing, and self-reflection |

## Configuration

Each database-dataset combination has a dedicated YAML config file. Below is an example showing all available sections:

```yaml
qdrant:
  host: "${QDRANT_HOST:-localhost}"
  port: "${QDRANT_PORT:-6333}"
  api_key: "${QDRANT_API_KEY:-}"

collection:
  name: "my_collection"

dataloader:
  type: "triviaqa"
  dataset_name: "trivia_qa"
  split: "test"
  limit: null

indexing:
  chunk_size: 512
  chunk_overlap: 50
  batch_size: 32

embeddings:
  model: "Qwen/Qwen3-Embedding-0.6B"
  batch_size: 32

generator:
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY:-}"
  max_tokens: 512
  temperature: 0.7

agentic_rag:
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY:-}"
  routing_enabled: true
  self_reflection_enabled: false
  max_iterations: 2
  quality_threshold: 75
  fallback_tool: "retrieval"
  max_retries: 3
  retry_delay_seconds: 0.5

retrieval:
  top_k_default: 10

evaluation:
  enabled: true
  metrics:
    - contextual_recall
    - contextual_precision
    - answer_relevancy
    - faithfulness

logging:
  name: "agentic_rag"
  level: "INFO"
```

## Directory Structure

```
agentic_rag/
├── __init__.py                      # Package exports for all pipeline classes
├── README.md                        # This file
├── base.py                          # Shared base class with routing, reflection, and evaluation
├── mixins.py                        # Retry and fallback mixin classes
├── utils.py                         # Batch iteration and helper utilities
├── chroma_agentic_rag.py            # Chroma database pipeline
├── milvus_agentic_rag.py            # Milvus database pipeline
├── pinecone_agentic_rag.py          # Pinecone database pipeline
├── qdrant_agentic_rag.py            # Qdrant database pipeline
├── weaviate_agentic_rag.py          # Weaviate database pipeline
└── configs/                         # 25 YAML configs (5 databases x 5 datasets)
    ├── chroma_arc.yaml
    ├── chroma_earnings_calls.yaml
    ├── chroma_factscore.yaml
    ├── chroma_popqa.yaml
    ├── chroma_triviaqa.yaml
    ├── milvus_arc.yaml
    ├── ...
    ├── qdrant_triviaqa.yaml
    ├── weaviate_arc.yaml
    └── weaviate_triviaqa.yaml
```

## Related Modules

- `src/vectordb/haystack/rag/` - Standard RAG pipelines without agentic routing or self-reflection
- `src/vectordb/haystack/diversity_filtering/` - Diversity-aware retrieval that can complement agentic RAG
- `src/vectordb/dataloaders/haystack/` - Dataset loaders for TriviaQA, ARC, PopQA, FactScore, and Earnings Calls
