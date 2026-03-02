# LangChain Integration

This module provides LangChain-based retrieval and RAG pipeline implementations across all five supported vector database backends. Every feature is organized as a self-contained directory with configuration files, indexing scripts, and search scripts for each backend.

## What You Get

- Seventeen retrieval and RAG patterns implemented using LangChain's retriever, chain, and document store abstractions.
- Full portability across Pinecone, Weaviate, Chroma, Milvus, and Qdrant (with feature-specific notes on backend support).
- YAML-driven configuration with environment variable substitution so credentials stay out of code.
- Evaluation support via the shared `utils/evaluation.py` metrics.
- Shared reusable components (`components/`) and helper factories (`utils/`) that all feature pipelines draw from.

## How the Module Is Structured

Each feature directory follows the same layout:

```
feature_name/
├── configs/
│   ├── chroma_triviaqa.yaml
│   ├── milvus_triviaqa.yaml
│   ├── pinecone_triviaqa.yaml
│   ├── qdrant_triviaqa.yaml
│   ├── weaviate_triviaqa.yaml
│   └── (one config per backend × dataset combination)
├── indexing/
│   ├── chroma.py
│   ├── milvus.py
│   ├── pinecone.py
│   ├── qdrant.py
│   └── weaviate.py
├── search/
│   ├── chroma.py
│   ├── milvus.py
│   ├── pinecone.py
│   ├── qdrant.py
│   └── weaviate.py
└── README.md
```

Indexing scripts load a dataset, embed documents using `EmbedderHelper`, and upsert them into the target backend. Search scripts embed a query, retrieve candidates, apply post-retrieval processing, and optionally generate an answer using `RAGHelper`.

## Feature Catalog

### `semantic_search/`

Dense vector similarity search. Documents are embedded at index time using `HuggingFaceEmbeddings` and stored in the vector backend. At query time, the same model embeds the query and the backend returns the most similar documents by cosine similarity. This is the baseline pattern.

### `hybrid_indexing/`

Combines dense semantic embeddings with sparse lexical embeddings. Both types are indexed per document. At query time, dense and sparse retrieval are run in parallel, and results are merged using Reciprocal Rank Fusion (RRF) via `ResultMerger.fuse_rrf()` from `utils/fusion.py`.

### `sparse_indexing/`

Sparse-only retrieval using token-weight vectors. Documents and queries are encoded using sparse embedding models. Performs well for exact terminology and keyword-driven queries but does not generalize to paraphrase or semantic variations.

### `reranking/`

Two-stage retrieval: the first stage retrieves a broad candidate pool using vector similarity, and the second stage reranks candidates using a HuggingFace cross-encoder model (`HuggingFaceCrossEncoder` from `langchain-community`). The cross-encoder scores each query-document pair jointly for higher-precision ranking. Created and applied via `RerankerHelper`.

### `mmr/`

Maximal Marginal Relevance diversity-aware reranking using `MMRHelper`. After initial retrieval, MMR iteratively selects documents that balance relevance to the query with dissimilarity from already-selected documents. The score formula is: `MMR(d) = λ × sim(d, query) − (1 − λ) × max_sim(d, selected)`. The `lambda_param` controls the tradeoff (1.0 = pure relevance, 0.0 = pure diversity).

### `diversity_filtering/`

Post-retrieval filtering using `helpers.py` to remove redundant documents from the candidate set. This uses cosine similarity among retrieved documents to identify and drop near-duplicates, giving the generator a more diverse context to work from.

### `metadata_filtering/`

Structured filter constraints applied at query time. Documents are indexed with metadata fields. Queries specify filter conditions that the backend applies to restrict the search space before scoring by embedding similarity. Filter dicts use MongoDB-style operators and are translated to backend-native formats via `FiltersHelper`.

### `json_indexing/`

Designed for JSON-native document corpora. Structured fields from JSON records are preserved as metadata and used for filter-based retrieval, while textual content is embedded for semantic search. Useful for product catalogs, APIs, event streams, and knowledge bases with mixed structured and unstructured content.

### `multi_tenancy/`

Tenant-scoped indexing and retrieval that ensures one tenant cannot access another's content. Uses `inject_scope_to_metadata()` at index time and `inject_scope_to_filter()` at query time. Backend-specific implementations use native isolation mechanisms: Chroma tenant/database context, Milvus partition keys, Weaviate tenants, Qdrant payload-based filtering, Pinecone namespaces.

### `namespaces/`

Logical data partitioning within a shared index. Queries are scoped to specific namespace partitions. Each backend implementation (`chroma.py`, `milvus.py`, `pinecone.py`, `qdrant.py`, `weaviate.py`) maps the namespace concept to the backend's native segmentation mechanism.

### `parent_document_retrieval/`

Indexes small child chunks for high-precision semantic matching, then maps retrieved children back to their parent documents to return more complete context. The `parent_store.py` file manages the parent-child ID mapping. At search time, retrieved child chunk IDs are resolved to their parent document text.

### `query_enhancement/`

Uses a Groq Llama LLM (`ChatGroq`) to generate improved queries before retrieval. The `QueryEnhancer` component implements three strategies:

- **Multi-query** (`generate_multi_queries`): Generates 5 alternative phrasings. Each is searched independently and results are deduplicated and merged.
- **HyDE** (`generate_hyde_queries`): Generates a hypothetical document answer, then uses it for retrieval alongside the original query.
- **Step-back** (`generate_step_back_queries`): Generates 3 broader background questions and searches with them plus the original, helping retrieve foundational context.

### `contextual_compression/`

Compresses retrieved documents to query-relevant fragments before passing context to the generator. The LangChain `ContextualCompressionRetriever` wraps any base retriever with a compressor. Implementations use LangChain-native compressor classes or custom `LLMChainExtractor` configurations.

### `cost_optimized_rag/`

Budget-aware retrieval with tunable controls on candidate pool size, context length passed to the generator, and model selection. Tracks cost-quality tradeoffs against evaluation datasets. Settings like `candidate_pool_size`, `context_budget`, and `model_tiering` give explicit levers to reduce spend while preserving acceptable quality.

### `agentic_rag/`

Multi-step iterative RAG using `AgenticRouter` (from `components/`). The pipeline state machine routes between three actions:

- **`search`**: Retrieve more documents from the vector store.
- **`reflect`**: Evaluate the current answer for gaps and inaccuracies.
- **`generate`**: Produce the final answer.

The router uses `ChatGroq` to make decisions at each step based on the current query, retrieved documents, and candidate answer. A hard `max_iterations` limit prevents infinite loops.

### `components/`

Reusable LangChain component classes:

- **`AgenticRouter`** (`agentic_router.py`): JSON-structured LLM routing using `ChatGroq` and `PromptTemplate`. Validates output structure and action values. Falls back to `"generate"` at iteration limit.
- **`ContextCompressor`** (`context_compressor.py`): LLM-based context compression using `ChatGroq`.
- **`QueryEnhancer`** (`query_enhancer.py`): Multi-query, HyDE, and step-back query expansion using `ChatGroq` and `PromptTemplate`.

### `utils/`

Shared helper classes used across all LangChain feature pipelines:

- **`EmbedderHelper`** (`embeddings.py`): Creates `HuggingFaceEmbeddings` from config, embeds document batches, and embeds individual queries.
- **`RerankerHelper`** (`reranker.py`): Creates `HuggingFaceCrossEncoder` from config and applies reranking with or without score preservation.
- **`MMRHelper`** (`mmr.py`): Pure-Python MMR algorithm using cosine similarity. Returns ranked `(Document, score)` tuples or just reranked documents.
- **`ResultMerger`** (`fusion.py`): RRF and weighted fusion with deduplication for hybrid search result merging.
- **`RAGHelper`** (`rag.py`): Creates `ChatGroq` LLM from config and formats retrieved documents into a prompt for answer generation.
- **`DocumentConverter`** (`document_converter.py`): Converts between LangChain `Document` objects and backend-native formats.
- **`FiltersHelper`** (`filters.py`): Translates MongoDB-style filter dicts to backend-native filter syntax.
- **`SparseEmbeddingHelper`** (`sparse_embeddings.py`): Creates and applies sparse embedding models for LangChain hybrid pipelines.
- **`DiversificationHelper`** (`diversification.py`): Cosine similarity-based near-duplicate filtering.
- **`ConfigLoader`** (`config.py`): Re-exports `ConfigLoader` from `vectordb.utils.config_loader`.

## Embedding Configuration

All LangChain feature pipelines read embedding configuration from the YAML config under the `embeddings` key:

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Required: full model path
  device: "cpu"                                       # Optional: "cpu" or "cuda"
  batch_size: 32                                      # Optional
```

For hybrid and sparse features, also include:

```yaml
sparse:
  model: "naver/splade-cocondenser-ensembledistil"   # Required for sparse embedder
```

## RAG Configuration

Generation is controlled by the `rag` section:

```yaml
rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.7
  max_tokens: 2048
```

The LangChain `RAGHelper` uses `ChatGroq` for generation. Set `enabled: false` to run retrieval-only pipelines.

## Recommended Onboarding Path

1. Run `semantic_search` on your target backend with a small dataset limit and verify the pipeline completes successfully.
2. Extract evaluation queries from the dataset and measure baseline retrieval metrics.
3. Add one improvement feature at a time — start with `reranking` (usually the highest single-step gain) or `hybrid_indexing` (for mixed query types).
4. Once quality is stable, layer in `multi_tenancy` or `namespaces` for data isolation.
5. Use `cost_optimized_rag` to find acceptable quality-cost tradeoffs, and `agentic_rag` for complex multi-step reasoning tasks.

## How to Choose a Feature

| If you need... | Use |
|---|---|
| Starting point and baseline | `semantic_search` |
| Both semantic and keyword precision | `hybrid_indexing` |
| Pure keyword/lexical precision | `sparse_indexing` |
| Better final ranking | `reranking` |
| Relevant + diverse result set | `mmr` |
| Less redundant context | `diversity_filtering` |
| Structured constraints | `metadata_filtering` |
| JSON-native documents | `json_indexing` |
| Better query recall | `query_enhancement` |
| Shorter, cleaner context | `contextual_compression` |
| Token/cost budget control | `cost_optimized_rag` |
| Iterative multi-step reasoning | `agentic_rag` |
| Long docs with fragment search | `parent_document_retrieval` |
| Per-customer data isolation | `multi_tenancy` |
| Logical data segmentation | `namespaces` |
