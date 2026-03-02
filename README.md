<h1 align="center"> <a href="https://github.com/avnlp/vectordb"> VectorDB </a> </h1>

<div align="center">

[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/avnlp/vectordb)
[![CI](https://img.shields.io/github/actions/workflow/status/avnlp/vectordb/ci.yml?branch=main&label=CI&logo=githubactions)](https://github.com/avnlp/vectordb/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/github/avnlp/vectordb/graph/badge.svg)](https://codecov.io/github/avnlp/vectordb)
[![Ruff](https://img.shields.io/github/actions/workflow/status/avnlp/vectordb/ci.yml?branch=main&label=Ruff&logo=ruff)](https://github.com/avnlp/vectordb/actions/workflows/ci.yml)
[![MyPy](https://img.shields.io/github/actions/workflow/status/avnlp/vectordb/ci.yml?branch=main&label=MyPy&logo=python)](https://github.com/avnlp/vectordb/actions/workflows/ci.yml)
[![Bandit](https://img.shields.io/github/actions/workflow/status/avnlp/vectordb/ci.yml?branch=main&label=Bandit&logo=owasp)](https://github.com/avnlp/vectordb/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/avnlp/vectordb/ci.yml?branch=main&label=Tests&logo=pytest)](https://github.com/avnlp/vectordb/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/avnlp/vectordb?color=green)](https://github.com/avnlp/vectordb/blob/main/LICENSE)

</div>

VectorDB provides a unified, production-oriented toolkit for Semantic Search and Retrieval-Augmented Generation across five vector databases, with feature parity between Haystack and LangChain.

It ships ready-to-run pipelines for Dense, Sparse, and Hybrid Retrieval, plus advanced RAG capabilities like Reranking, Query Enhancement, Contextual Compression, Parent-Child Retrieval, and Agentic Retrieval Loops. The design is configuration-driven, environment-variable friendly, and built for consistent benchmarking across databases and datasets. Use it to build, compare, and deploy retrieval systems without re-implementing logic per backend.

## Vector Databases

- **Pinecone**: Managed vector database with namespaces and native sparse-dense hybrid retrieval.
- **Weaviate**: Open-source vector search with BM25 hybrid retrieval, collections, and multi-tenancy.
- **Qdrant**: High-performance search with payload filtering and scalar or binary quantization.
- **Milvus**: Scalable vector database with partition-key isolation and hybrid retrieval.
- **Chroma**: Lightweight vector store for local development and rapid prototyping.

## Datasets & Evaluation

VectorDB includes dataset loaders and standardized evaluation utilities so you can benchmark retrieval quality across databases and frameworks.

Supported datasets:

- **TriviaQA** - Open-domain question-answer pairs for general knowledge retrieval.
- **ARC** - Science reasoning questions requiring multi-hop inference.
- **PopQA** - Factoid questions about popular entities.
- **FactScore** - Atomic facts for verification and hallucination detection.
- **Earnings Calls** - Financial transcript Q&A for domain-specific RAG.

Built-in evaluation metrics:

- Recall@k
- Precision@k
- MRR
- NDCG@k
- Hit rate

## Features

| Feature | What it enables |
|:--|:--|
| **Semantic Search** | Dense vector retrieval with metadata filters and optional answer generation. |
| **Sparse Search** | Keyword-focused retrieval using sparse encoders for exact terminology. |
| **Hybrid Search** | Dense + sparse retrieval fused with RRF or weighted scoring. |
| **Reranking** | Two-stage retrieval using cross-encoders or API rerankers for higher precision. |
| **MMR Diversity** | Maximal marginal relevance to balance relevance and diversity. |
| **Diversity Filtering** | Remove near-duplicate results using similarity or clustering. |
| **Metadata Filtering** | Structured filtering on fields and nested JSON paths. |
| **JSON Indexing** | Index and query structured JSON documents with path-based filters. |
| **Query Enhancement** | Multi-query, HyDE, and step-back prompting to improve recall. |
| **Contextual Compression** | Reduce retrieved context via reranking or LLM extraction. |
| **Parent Document Retrieval** | Index chunks but return parent documents or context windows. |
| **Namespaces** | Logical partitioning for environment separation and dataset versioning. |
| **Multi-Tenancy** | Tenant isolation using database-specific strategies at scale. |
| **Cost-Optimized RAG** | Hybrid retrieval with local sparse embeddings and optional generation to reduce API cost. |
| **Agentic RAG** | Iterative retrieval loop with search, reflect, and refine steps. |

### Semantic Search (Dense Search)

Semantic Search converts text into high-dimensional vectors using transformer embedding models, then finds documents with similar vector representations. This approach understands synonyms, paraphrases, and conceptual similarity—queries like "car" will match documents about "automobile" even without exact keyword overlap.

- Supports any SentenceTransformers model
- Optional semantic diversification removes near-duplicate results
- Integrates with Groq or OpenAI for RAG answer generation
- Metadata filters narrow results by category, date, source, or custom fields

### Sparse Search (Keyword Search)

Sparse Search uses SPLADE models to create sparse vectors that emphasize specific terms, similar to traditional BM25 but with learned term importance. This excels when exact terminology matters—legal documents, product SKUs, or technical specifications.

- Supports SPLADE-based or BM25-style sparse encoders
- Weaviate uses native BM25 without external embeddings
- Works alongside dense search or as a standalone retrieval method

### Hybrid Search

Hybrid search runs both dense and sparse retrieval in parallel, then fuses results using Reciprocal Rank Fusion (RRF) or weighted combination. You get the best of both worlds: semantic understanding for concepts plus keyword precision for specific terms.

- Dense + sparse fusion with configurable weights
- RRF handles score normalization automatically
- Built-in evaluation metrics: Recall@k, MRR, NDCG, Precision@k
- No FastEmbed dependency—uses native SentenceTransformers sparse encoders

### Metadata Filtering

Filter search results using structured document attributes before or after vector search. A product catalog might filter by `category = "electronics"` and `price < 500` while still ranking by semantic relevance.

- Operators: equals, not equals, greater than, less than, in list, contains, range
- Database-specific expression builders generate optimal filter syntax
- Selectivity analysis helps optimize filter order for performance
- Timing metrics track pre-filter vs. vector search latency

### Reranking

Two-stage retrieval first casts a wide net with fast vector search, then uses a cross-encoder model to precisely score the top candidates. Cross-encoders see query and document together, enabling much finer relevance judgments than embedding similarity alone.

- Models: modern cross-encoder rerankers and lightweight scoring models
- Integrated evaluation with contextual recall, precision, and faithfulness metrics

### MMR (Maximal Marginal Relevance)

MMR balances relevance with diversity by penalizing documents too similar to those already selected. The result set covers more aspects of a topic instead of repeating similar content.

- Tune relevance vs. diversity to fit the task
- Uses SentenceTransformers DiversityRanker component
- Particularly useful for summarization and exploratory search

### Namespaces

Namespaces partition data within a single index, enabling logical separation without managing multiple collections. Use cases include separating development from production data, versioning document sets, or organizing by content type.

- Pinecone: native namespace support
- Milvus: partition-based isolation
- Qdrant, Chroma, Weaviate: collection-based separation
- Cross-namespace search queries multiple partitions simultaneously

### Multi-Tenancy

Multi-tenancy isolates customer data so each tenant only sees their own documents. Isolation
strategies include namespaces, partitions, payload filters, and tenant-scoped collections,
tailored to the selected vector database.

| Database | Isolation Strategy | Scale |
|-|-|-|
| Milvus | Partition key with filter expressions | Millions of tenants |
| Weaviate | Native multi-tenancy with per-tenant shards | Enterprise-grade |
| Pinecone | Namespace-based isolation | 100,000+ tenants |
| Qdrant | Payload-based with optimized indexes | Tiered promotion |
| Chroma | Tenant and database scoping | Flexible |

### Query Enhancement

Query enhancement rewrites user queries to improve retrieval. Three techniques address different challenges:

- **Multi-Query** generates N variations of the original query, retrieves for each, and fuses results. Handles ambiguous or underspecified questions.
- **HyDE (Hypothetical Document Embeddings)** asks an LLM to write a hypothetical answer, then searches for documents similar to that answer. Bridges the vocabulary gap between questions and documents.
- **Step-Back Prompting** generates a more abstract version of the query to retrieve broader context before answering specific questions.

### Parent Document Retrieval

Index small chunks for precise matching, but return the larger parent document for context. This solves the chunk-size tradeoff: small chunks match accurately, but lack the surrounding context needed for good answers.

- Retrieval modes: children only, with parents, or context window
- Configurable parent and child chunk sizes with overlap
- In-memory parent store links chunks to their source documents

### Contextual Compression

Reduce retrieved context to only the most relevant passages, cutting token costs for LLM generation. Two approaches trade off speed versus precision:

- **Reranking-based** uses cross-encoders to score and filter passages (fast, nearly free)
- **LLM extraction** asks a model to extract only relevant sentences (slower, more precise)

Metrics track compression ratio and tokens saved per query.

### Cost-Optimized RAG

Production RAG pipelines need cost controls. This module provides:

- **Pre-filtering** narrows the search space before vector operations
- **Batch processing** groups multiple queries for efficient embedding and search
- **Result caching** with LRU eviction avoids repeated searches
- **Cost monitoring** tracks API calls, tokens, and estimated costs per operation

### Agentic RAG

Agentic RAG introduces a decision-making loop where an LLM agent controls the retrieval process. Instead of a fixed pipeline, the agent chooses actions based on what it learns:

- **Route** — Decide whether to search, reflect, or generate based on current state
- **Search** — Retrieve relevant documents from the vector database
- **Compress** — Extract the most relevant passages from retrieved documents
- **Reflect** — Evaluate answer quality and decide whether to iterate
- **Generate** — Produce the final answer when confident

The agent can iterate multiple times, refining its search strategy based on what it finds. This handles complex questions that require multiple retrieval steps or benefit from self-correction.

### JSON Indexing

Index and search structured JSON documents with nested field support. Query by JSON paths while still leveraging vector similarity for the text content.

### Diversity Filtering

Remove redundant documents using clustering or embedding similarity. When search returns many near-duplicates, diversity filtering selects representative documents to maximize information coverage.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management. First, ensure uv is installed:

```bash
# Install uv (if not already installed)
pip install uv
```

Then install the project dependencies:

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Quick Start

For detailed usage examples, see the feature-level READMEs under `src/vectordb/`.

### Semantic Search (Haystack)

```python
from vectordb.haystack.semantic_search import PineconeSemanticSearchPipeline

pipeline = PineconeSemanticSearchPipeline(
    "src/vectordb/haystack/semantic_search/configs/pinecone/arc.yaml"
)
result = pipeline.search("What is photosynthesis?", top_k=5)

for doc in result["documents"]:
    print(doc.content)
```

### Hybrid Search (Haystack)

```python
from vectordb.haystack.hybrid_indexing import MilvusHybridSearchPipeline

pipeline = MilvusHybridSearchPipeline(
    "src/vectordb/haystack/hybrid_indexing/configs/milvus_triviaqa.yaml"
)
result = pipeline.run(query="machine learning algorithms", top_k=10)
```

### Semantic Search (LangChain)

```python
from vectordb.langchain.semantic_search import WeaviateSemanticSearchPipeline

pipeline = WeaviateSemanticSearchPipeline(
    "src/vectordb/langchain/semantic_search/configs/weaviate_popqa.yaml"
)
result = pipeline.search("Who invented the telephone?", top_k=5)
```

### Agentic RAG (LangChain)

```python
from vectordb.langchain.agentic_rag import PineconeAgenticRAGPipeline

pipeline = PineconeAgenticRAGPipeline(
    "src/vectordb/langchain/agentic_rag/configs/pinecone_triviaqa.yaml"
)
result = pipeline.run(query="Explain how neural networks learn", top_k=10)

print(result["final_answer"])
```

## Documentation

Comprehensive documentation is available under [`docs/`](docs/README.md), including:

- Core architecture and configuration references
- Framework-specific guides for Haystack and LangChain
- Feature-level conceptual and API-oriented documentation

## Contributing

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
