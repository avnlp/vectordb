# Haystack Integration

This module provides Haystack-based retrieval and RAG pipeline implementations across all five supported vector database backends. Every feature is organized as a self-contained directory with configuration files, indexing scripts, and search scripts for each backend.

## What You Get

- Seventeen retrieval and RAG patterns implemented using Haystack's pipeline and component abstractions.
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

Indexing scripts load a dataset, embed documents, and upsert them into the target backend. Search scripts embed a query, retrieve candidates, apply any post-retrieval processing (reranking, compression, filtering), and optionally generate an answer using the RAG generator.

## Feature Catalog

### `semantic_search/`

Dense vector similarity search. Documents and queries are embedded with the same SentenceTransformers model and compared by cosine similarity. This is the baseline pattern that all other features build on. Use this first to establish quality and latency benchmarks.

### `hybrid_indexing/`

Combines dense semantic embeddings with sparse lexical embeddings (SPLADE-style) to handle both natural-language queries and keyword-precise queries. Documents are indexed with both vector types. At query time, results from both retrievers are merged using Reciprocal Rank Fusion (RRF) via `ResultMerger.fuse_rrf()`.

### `sparse_indexing/`

Sparse-only retrieval using token-weight vectors. Documents are encoded using a sparse SentenceTransformers model. This is the lexical complement to semantic search, performing well for exact terminology and domain jargon but poorly for paraphrase-heavy queries.

### `reranking/`

Two-stage retrieval: a fast first-pass vector search retrieves a large candidate pool, then a cross-encoder model (`SentenceTransformersSimilarityRanker`) reranks the candidates by scoring query-document pairs jointly. Created via `RerankerFactory.create(config)`.

### `mmr/`

Maximal Marginal Relevance reranking using Haystack's `SentenceTransformersDiversityRanker`. After an initial retrieval pass, MMR iteratively selects documents that are both relevant to the query and dissimilar to already-selected documents. The `lambda` parameter controls the relevance-vs-diversity tradeoff. Created via `RerankerFactory.create_diversity_ranker(config)`.

### `diversity_filtering/`

Similarity count filtering implemented in `DiversificationHelper`. Unlike MMR (which globally optimizes relevance minus redundancy), this filter applies a local rule: a document is kept only if fewer than `max_similar_docs` already-selected documents have cosine similarity ≥ `diversity_threshold` with it. This preserves original relevance order while removing near-duplicates.

### `metadata_filtering/`

Structured metadata constraints applied before or alongside vector similarity search. Documents are indexed with metadata fields (category, date, entity type, etc.). Queries specify filter expressions that the backend uses to restrict the search space. Each backend uses its own filter syntax, which the wrapper translates from a common MongoDB-style dict format.

### `json_indexing/`

Designed for JSON-native documents with structured fields. Embeds textual content while preserving structured metadata fields for filter-based retrieval. Useful for APIs, events, product catalogs, or any corpus where both semantic content and structured attributes drive retrieval.

### `multi_tenancy/`

Tenant-scoped indexing and querying that prevents one tenant from accessing another's content. Tenant context is injected into document metadata at index time using `inject_scope_to_metadata()` and into query filters at search time using `inject_scope_to_filter()`. Backend-specific isolation mechanisms are used where available (Milvus partition keys, Weaviate tenants, Qdrant `is_tenant` payload indexes).

### `namespaces/`

Logical segmentation within a shared index or collection. Queries are scoped to one or more namespace partitions. This is lighter-weight than full multi-tenancy — suitable for environment separation (prod/staging), versioned indexes, or customer groups where strict security isolation is not required.

### `parent_document_retrieval/`

Indexes small child chunks for precise semantic matching, then maps retrieved children back to their parent documents to return richer, more coherent context. Child-to-parent relationships are stored in metadata. The search phase deduplicates parents when multiple children from the same parent are retrieved.

### `query_enhancement/`

Uses an LLM to generate improved retrieval queries before searching. Three strategies are supported via `QueryEnhancer`:

- **Multi-query**: Generates N alternative phrasings of the original query. All versions are searched and results are merged.
- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical answer document and uses it for retrieval, bridging the query-document distribution gap.
- **Step-back**: Generates a broader, more abstract version of the query to retrieve relevant background context.

### `contextual_compression/`

Reduces retrieved context to the most query-relevant fragments using `ContextCompressor`. Three compression strategies:

- **Abstractive**: LLM-generated summary of retrieved content focused on the query.
- **Extractive**: Selects the most relevant sentences from the retrieved text.
- **Relevance filtering**: Removes paragraphs with relevance below a configurable threshold.

All compression methods fall back to the original context on failure.

### `cost_optimized_rag/`

Controls compute and token spend across retrieval and generation stages. Achieves cost reduction by tuning the retrieval breadth (smaller candidate pools), applying compression to limit generation input size, and selecting cheaper model tiers for less critical pipeline stages.

### `agentic_rag/`

Multi-step iterative RAG with self-reflection and query reformulation using `AgenticRouter`. The pipeline loop:

1. Retrieves an initial set of documents.
2. Generates a draft answer.
3. Evaluates answer quality (relevance, completeness, grounding).
4. Decides whether to refine (reformulate and retrieve again) or finalize.
5. Returns the final answer after the loop converges or hits `max_iterations`.

Tool selection (retrieval, web search, calculation, reasoning) is also supported for routing different query types to different pipeline branches.

### `components/`

Reusable Haystack component classes:

- **`AgenticRouter`**: LLM-based tool selector and self-reflection loop for agentic RAG. Uses Haystack's `OpenAIChatGenerator` with a Groq endpoint.
- **`ContextCompressor`**: Three compression strategies (abstractive, extractive, relevance filter) using Haystack's chat generator.
- **`QueryEnhancer`**: Multi-query, HyDE, and step-back query expansion using an LLM.
- **`ResultMerger`**: RRF and weighted score fusion for hybrid search results (from `utils/fusion.py`, re-exported here).

### `utils/`

Shared helper factories and wrappers used across all Haystack feature pipelines:

- **`EmbedderFactory`** (`embeddings.py`): Creates and warms up `SentenceTransformersDocumentEmbedder`, `SentenceTransformersTextEmbedder`, `SentenceTransformersSparseDocumentEmbedder`, and `SentenceTransformersSparseTextEmbedder` from config dicts. Also probes embedding dimension by running a sample document.
- **`RerankerFactory`** (`reranker.py`): Creates `SentenceTransformersSimilarityRanker` (cross-encoder) and `SentenceTransformersDiversityRanker` (MMR) from config dicts, with automatic warm-up.
- **`ResultMerger`** (`fusion.py`): RRF and weighted fusion for hybrid search. Both methods deduplicate by document ID.
- **`DiversificationHelper`** (`diversification.py`): Sequential similarity count filter for deduplication.
- **`RAGHelper`** (`rag.py`): Creates a Groq-compatible `OpenAIGenerator` from config and formats retrieved documents into a prompt for generation.
- **`ConfigLoader`** (`config.py`): Re-exports `ConfigLoader` from `vectordb.utils.config_loader`.
- **`FiltersHelper`** (`filters.py`): Converts MongoDB-style filter dicts to backend-native filter objects or expressions for Haystack document stores.

## Embedding Configuration

All Haystack feature pipelines read embedding configuration from the YAML config under the `embeddings` key:

```yaml
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Required: full model path
  device: "cpu"                                       # Optional: "cpu" or "cuda"
  batch_size: 32                                      # Optional: document embedding batch size
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
  api_base_url: "https://api.groq.com/openai/v1"
  temperature: 0.7
  max_tokens: 2048
```

Set `enabled: false` to run retrieval-only pipelines without generation. When `enabled: true`, the `GROQ_API_KEY` environment variable must be set.

## Recommended Onboarding Path

1. Run `semantic_search` on your target backend with a small dataset limit (100–200 records) and verify that the pipeline loads, indexes, and retrieves successfully.
2. Measure retrieval quality using `evaluation_queries()` and `evaluate_retrieval()`.
3. Add one improvement feature at a time (for example, `reranking` or `hybrid_indexing`) and measure whether quality improves on your evaluation set.
4. Once the retrieval baseline is strong, adopt `multi_tenancy` or `namespaces` for data isolation, and `cost_optimized_rag` for budget controls.
5. Use `agentic_rag` or `query_enhancement` for hard multi-hop questions where single-pass retrieval falls short.

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
