# Public API Reference

## 1. What This Feature Is

This document provides a **complete inventory of all public APIs** exported by the VectorDB package. It serves as the authoritative reference for:

- **Import paths**: Where to import classes/functions from
- **Class hierarchies**: Inheritance and composition relationships
- **Method signatures**: Parameters and return types
- **Type definitions**: Data classes and enums

## 2. Package Structure

```
vectordb/
├── databases/          # Backend wrappers
├── dataloaders/        # Dataset loading
├── haystack/           # Haystack integrations
├── langchain/          # LangChain integrations
└── utils/              # Shared utilities
```

## 3. Core APIs (`vectordb.*`)

### Database Wrappers (`vectordb.databases`)

```python
from vectordb.databases import (
    ChromaVectorDB,
    MilvusVectorDB,
    PineconeVectorDB,
    QdrantVectorDB,
    WeaviateVectorDB,
)
```

**Common Methods**:

- `create_collection(name, dimension, ...)`
- `upsert(documents)` / `insert_documents(documents)`
- `search(query_embedding, top_k, filters)`
- `query(query_embedding, n_results, where)`
- `delete(ids)` / `delete_collection(name)`
- `list_collections()`

### Dataloaders (`vectordb.dataloaders`)

```python
from vectordb.dataloaders import (
    DataloaderCatalog,
    LoadedDataset,
    DatasetRecord,
    EvaluationQuery,
    DocumentConverter,
)
```

**Key Classes**:

- `DataloaderCatalog.create(name, split, limit)` → Loader instance
- `LoadedDataset.records()` → `List[DatasetRecord]`
- `LoadedDataset.to_haystack()` → `List[Document]`
- `LoadedDataset.to_langchain()` → `List[Document]`
- `LoadedDataset.evaluation_queries(limit)` → `List[EvaluationQuery]`

### Utilities (`vectordb.utils`)

```python
from vectordb.utils import (
    load_config,
    resolve_env_vars,
    RetrievalMetrics,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    ChromaDocumentConverter,
    PineconeDocumentConverter,
    QdrantDocumentConverter,
    WeaviateDocumentConverter,
    get_doc_id,
    coerce_id,
    set_doc_id,
    inject_scope_to_metadata,
    inject_scope_to_filter,
    normalize_sparse,
    to_milvus_sparse,
    to_pinecone_sparse,
    to_qdrant_sparse,
    LoggerFactory,
)
```

## 4. Haystack APIs (`vectordb.haystack.*`)

### Semantic Search

```python
from vectordb.haystack.semantic_search import (
    ChromaSemanticIndexingPipeline,
    MilvusSemanticIndexingPipeline,
    PineconeSemanticIndexingPipeline,
    QdrantSemanticIndexingPipeline,
    WeaviateSemanticIndexingPipeline,
    ChromaSemanticSearchPipeline,
    MilvusSemanticSearchPipeline,
    PineconeSemanticSearchPipeline,
    QdrantSemanticSearchPipeline,
    WeaviateSemanticSearchPipeline,
)
```

### Hybrid Indexing

```python
from vectordb.haystack.hybrid_indexing import (
    ChromaHybridIndexingPipeline,
    MilvusHybridIndexingPipeline,
    PineconeHybridIndexingPipeline,
    QdrantHybridIndexingPipeline,
    WeaviateHybridIndexingPipeline,
    ChromaHybridSearchPipeline,
    MilvusHybridSearchPipeline,
    PineconeHybridSearchPipeline,
    QdrantHybridSearchPipeline,
    WeaviateHybridSearchPipeline,
)
```

### Sparse Indexing

```python
from vectordb.haystack.sparse_indexing import (
    ChromaSparseIndexingPipeline,
    MilvusSparseIndexingPipeline,
    PineconeSparseIndexingPipeline,
    QdrantSparseIndexingPipeline,
    WeaviateBM25IndexingPipeline,
    ChromaSparseSearchPipeline,
    MilvusSparseSearchPipeline,
    PineconeSparseSearchPipeline,
    QdrantSparseSearchPipeline,
    WeaviateBM25SearchPipeline,
)
```

### Reranking

```python
from vectordb.haystack.reranking import (
    ChromaRerankingIndexingPipeline,
    MilvusRerankingIndexingPipeline,
    PineconeRerankingIndexingPipeline,
    QdrantRerankingIndexingPipeline,
    WeaviateRerankingIndexingPipeline,
    ChromaRerankingSearchPipeline,
    MilvusRerankingSearchPipeline,
    PineconeRerankingSearchPipeline,
    QdrantRerankingSearchPipeline,
    WeaviateRerankingSearchPipeline,
)
```

### MMR

```python
from vectordb.haystack.mmr import (
    ChromaMmrIndexingPipeline,
    MilvusMmrIndexingPipeline,
    PineconeMmrIndexingPipeline,
    QdrantMmrIndexingPipeline,
    WeaviateMmrIndexingPipeline,
    ChromaMmrSearchPipeline,
    MilvusMmrSearchPipeline,
    PineconeMmrSearchPipeline,
    QdrantMmrSearchPipeline,
    WeaviateMmrSearchPipeline,
)
```

### Metadata Filtering

```python
from vectordb.haystack.metadata_filtering import (
    ChromaMetadataFilteringIndexingPipeline,
    MilvusMetadataFilteringIndexingPipeline,
    PineconeMetadataFilteringIndexingPipeline,
    QdrantMetadataFilteringIndexingPipeline,
    WeaviateMetadataFilteringIndexingPipeline,
    ChromaMetadataFilteringSearchPipeline,
    MilvusMetadataFilteringSearchPipeline,
    PineconeMetadataFilteringSearchPipeline,
    QdrantMetadataFilteringSearchPipeline,
    WeaviateMetadataFilteringSearchPipeline,
)
```

### Query Enhancement

```python
from vectordb.haystack.query_enhancement import (
    ChromaQueryEnhancementIndexingPipeline,
    MilvusQueryEnhancementIndexingPipeline,
    PineconeQueryEnhancementIndexingPipeline,
    QdrantQueryEnhancementIndexingPipeline,
    WeaviateQueryEnhancementIndexingPipeline,
    ChromaQueryEnhancementSearchPipeline,
    MilvusQueryEnhancementSearchPipeline,
    PineconeQueryEnhancementSearchPipeline,
    QdrantQueryEnhancementSearchPipeline,
    WeaviateQueryEnhancementSearchPipeline,
)
```

### Parent Document Retrieval

```python
from vectordb.haystack.parent_document_retrieval import (
    ChromaParentDocIndexingPipeline,
    MilvusParentDocIndexingPipeline,
    PineconeParentDocIndexingPipeline,
    QdrantParentDocIndexingPipeline,
    WeaviateParentDocIndexingPipeline,
    ChromaParentDocSearchPipeline,
    MilvusParentDocSearchPipeline,
    PineconeParentDocSearchPipeline,
    QdrantParentDocSearchPipeline,
    WeaviateParentDocSearchPipeline,
)
```

### Contextual Compression

```python
from vectordb.haystack.contextual_compression import (
    BaseContextualCompressionPipeline,
    ChromaCompressionSearch,
    MilvusCompressionSearch,
    PineconeCompressionSearch,
    QdrantCompressionSearch,
    WeaviateCompressionSearch,
    CompressorFactory,
    TokenCounter,
)
```

### Contextual Compression Indexing

```python
from vectordb.haystack.contextual_compression.indexing import (
    BaseIndexingPipeline,
    ChromaIndexingPipeline,
    MilvusIndexingPipeline,
    PineconeIndexingPipeline,
    QdrantIndexingPipeline,
    WeaviateIndexingPipeline,
)
```

### Cost-Optimized RAG

```python
from vectordb.haystack.cost_optimized_rag import (
    ChromaIndexingPipeline,
    MilvusIndexingPipeline,
    PineconeIndexingPipeline,
    QdrantIndexingPipeline,
    WeaviateIndexingPipeline,
    ChromaSearchPipeline,
    MilvusSearchPipeline,
    PineconeSearchPipeline,
    QdrantSearchPipeline,
    WeaviateSearchPipeline,
)
```

### Diversity Filtering

```python
from vectordb.haystack.diversity_filtering import (
    run_indexing,
    run_search,
    ChromaDiversitySearchPipeline,
    ClusteringDiversityRanker,
)
```

### Agentic RAG

```python
from vectordb.haystack.agentic_rag import (
    BaseAgenticRAGPipeline,
    ChromaAgenticRAGPipeline,
    MilvusAgenticRAGPipeline,
    PineconeAgenticRAGPipeline,
    QdrantAgenticRAGPipeline,
    WeaviateAgenticRAGPipeline,
)
```

### Multi-Tenancy

```python
from vectordb.haystack.multi_tenancy import (
    BaseMultitenancyPipeline,
    ChromaMultitenancyIndexingPipeline,
    MilvusMultitenancyIndexingPipeline,
    PineconeMultitenancyIndexingPipeline,
    QdrantMultitenancyIndexingPipeline,
    WeaviateMultitenancyIndexingPipeline,
    ChromaMultitenancySearchPipeline,
    MilvusMultitenancySearchPipeline,
    PineconeMultitenancySearchPipeline,
    QdrantMultitenancySearchPipeline,
    WeaviateMultitenancySearchPipeline,
    TenantContext,
    TenantIsolationStrategy,
    TenantIndexResult,
    TenantRetrievalResult,
    TenantRAGResult,
    TenantQueryResult,
    TenantStats,
    TenantStatus,
)
```

### Namespaces

```python
from vectordb.haystack.namespaces import (
    PineconeNamespacePipeline,
    WeaviateNamespacePipeline,
    MilvusNamespacePipeline,
    QdrantNamespacePipeline,
    ChromaNamespacePipeline,
    IsolationStrategy,
    TenantStatus,
    NamespaceConfig,
    NamespaceStats,
    NamespaceTimingMetrics,
    NamespaceQueryResult,
    CrossNamespaceComparison,
    CrossNamespaceResult,
    NamespaceOperationResult,
    NamespaceError,
    NamespaceNotFoundError,
    NamespaceExistsError,
    NamespaceConnectionError,
    NamespaceOperationNotSupportedError,
    Timer,
    NamespaceNameGenerator,
    QuerySampler,
)
```

### Components

```python
from vectordb.haystack.components import (
    AgenticRouter,
    QueryEnhancer,
    ContextCompressor,
    ResultMerger,
    DeepEvalEvaluator,
)
```

### Utils

```python
from vectordb.haystack.utils import (
    ConfigLoader,
    DiversificationHelper,
    EmbedderFactory,
    DocumentFilter,
    RAGHelper,
    RerankerFactory,
    ResultMerger,
)
```

## 5. LangChain APIs (`vectordb.langchain.*`)

### Semantic Search

```python
from vectordb.langchain.semantic_search import (
    ChromaSemanticIndexingPipeline,
    MilvusSemanticIndexingPipeline,
    PineconeSemanticIndexingPipeline,
    QdrantSemanticIndexingPipeline,
    WeaviateSemanticIndexingPipeline,
    ChromaSemanticSearchPipeline,
    MilvusSemanticSearchPipeline,
    PineconeSemanticSearchPipeline,
    QdrantSemanticSearchPipeline,
    WeaviateSemanticSearchPipeline,
)
```

### Hybrid Indexing

```python
from vectordb.langchain.hybrid_indexing import (
    ChromaHybridIndexingPipeline,
    MilvusHybridIndexingPipeline,
    PineconeHybridIndexingPipeline,
    QdrantHybridIndexingPipeline,
    WeaviateHybridIndexingPipeline,
    ChromaHybridSearchPipeline,
    MilvusHybridSearchPipeline,
    PineconeHybridSearchPipeline,
    QdrantHybridSearchPipeline,
    WeaviateHybridSearchPipeline,
)
```

### Sparse Indexing

```python
from vectordb.langchain.sparse_indexing import (
    ChromaSparseIndexingPipeline,
    MilvusSparseIndexingPipeline,
    PineconeSparseIndexingPipeline,
    QdrantSparseIndexingPipeline,
    WeaviateBM25IndexingPipeline,
    ChromaSparseSearchPipeline,
    MilvusSparseSearchPipeline,
    PineconeSparseSearchPipeline,
    QdrantSparseSearchPipeline,
    WeaviateBM25SearchPipeline,
)
```

### Reranking

```python
from vectordb.langchain.reranking import (
    ChromaRerankingIndexingPipeline,
    MilvusRerankingIndexingPipeline,
    PineconeRerankingIndexingPipeline,
    QdrantRerankingIndexingPipeline,
    WeaviateRerankingIndexingPipeline,
    ChromaRerankingSearchPipeline,
    MilvusRerankingSearchPipeline,
    PineconeRerankingSearchPipeline,
    QdrantRerankingSearchPipeline,
    WeaviateRerankingSearchPipeline,
)
```

### MMR

```python
from vectordb.langchain.mmr import (
    ChromaMmrIndexingPipeline,
    MilvusMmrIndexingPipeline,
    PineconeMmrIndexingPipeline,
    QdrantMmrIndexingPipeline,
    WeaviateMmrIndexingPipeline,
    ChromaMmrSearchPipeline,
    MilvusMmrSearchPipeline,
    PineconeMmrSearchPipeline,
    QdrantMmrSearchPipeline,
    WeaviateMmrSearchPipeline,
)
```

### Metadata Filtering

```python
from vectordb.langchain.metadata_filtering import (
    ChromaMetadataFilteringIndexingPipeline,
    MilvusMetadataFilteringIndexingPipeline,
    PineconeMetadataFilteringIndexingPipeline,
    QdrantMetadataFilteringIndexingPipeline,
    WeaviateMetadataFilteringIndexingPipeline,
    ChromaMetadataFilteringSearchPipeline,
    MilvusMetadataFilteringSearchPipeline,
    PineconeMetadataFilteringSearchPipeline,
    QdrantMetadataFilteringSearchPipeline,
    WeaviateMetadataFilteringSearchPipeline,
)
```

### Query Enhancement

```python
from vectordb.langchain.query_enhancement import (
    ChromaQueryEnhancementIndexingPipeline,
    MilvusQueryEnhancementIndexingPipeline,
    PineconeQueryEnhancementIndexingPipeline,
    QdrantQueryEnhancementIndexingPipeline,
    WeaviateQueryEnhancementIndexingPipeline,
    ChromaQueryEnhancementSearchPipeline,
    MilvusQueryEnhancementSearchPipeline,
    PineconeQueryEnhancementSearchPipeline,
    QdrantQueryEnhancementSearchPipeline,
    WeaviateQueryEnhancementSearchPipeline,
)
```

### Parent Document Retrieval

```python
from vectordb.langchain.parent_document_retrieval import (
    ChromaParentDocIndexingPipeline,
    MilvusParentDocIndexingPipeline,
    PineconeParentDocIndexingPipeline,
    QdrantParentDocIndexingPipeline,
    WeaviateParentDocIndexingPipeline,
    ChromaParentDocSearchPipeline,
    MilvusParentDocSearchPipeline,
    PineconeParentDocSearchPipeline,
    QdrantParentDocSearchPipeline,
    WeaviateParentDocSearchPipeline,
)
```

### Contextual Compression

```python
from vectordb.langchain.contextual_compression import (
    BaseContextualCompressionPipeline,
    ChromaCompressionSearch,
    MilvusCompressionSearch,
    PineconeCompressionSearch,
    QdrantCompressionSearch,
    WeaviateCompressionSearch,
    TokenCounter,
)
```

### Cost-Optimized RAG

```python
from vectordb.langchain.cost_optimized_rag import (
    ChromaIndexingPipeline,
    MilvusIndexingPipeline,
    PineconeIndexingPipeline,
    QdrantIndexingPipeline,
    WeaviateIndexingPipeline,
    ChromaSearchPipeline,
    MilvusSearchPipeline,
    PineconeSearchPipeline,
    QdrantSearchPipeline,
    WeaviateSearchPipeline,
    ResultFuser,
    RetrievalMetrics,
    MetricsAggregator,
)
```

### Diversity Filtering

```python
from vectordb.langchain.diversity_filtering import (
    run_indexing,
    run_search,
    ChromaDiversitySearchPipeline,
    ClusteringDiversityRanker,
)
```

### Agentic RAG

```python
from vectordb.langchain.agentic_rag import (
    BaseAgenticRAGPipeline,
    ChromaAgenticRAGPipeline,
    MilvusAgenticRAGPipeline,
    PineconeAgenticRAGPipeline,
    QdrantAgenticRAGPipeline,
    WeaviateAgenticRAGPipeline,
)
```

### Namespaces

```python
from vectordb.langchain.namespaces import (
    PineconeNamespacePipeline,
    WeaviateNamespacePipeline,
    MilvusNamespacePipeline,
    QdrantNamespacePipeline,
    ChromaNamespacePipeline,
    IsolationStrategy,
    TenantStatus,
    NamespaceConfig,
    NamespaceStats,
    NamespaceTimingMetrics,
    NamespaceQueryResult,
    CrossNamespaceComparison,
    CrossNamespaceResult,
    NamespaceOperationResult,
    NamespaceError,
    NamespaceNotFoundError,
    NamespaceExistsError,
    NamespaceConnectionError,
    NamespaceOperationNotSupportedError,
    Timer,
    NamespaceNameGenerator,
    QuerySampler,
)
```

### Components

```python
from vectordb.langchain.components import (
    AgenticRouter,
    QueryEnhancer,
    ContextCompressor,
)
```

### Utils

```python
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
    RerankerHelper,
    MMRHelper,
    ResultMerger,
    RAGHelper,
    DocumentConverter,
    FiltersHelper,
    DiversificationHelper,
)
```

## 6. Type Definitions

### Dataloaders

```python
from vectordb.dataloaders.types import (
    DatasetType,          # Literal["triviaqa", "arc", "popqa", "factscore", "earnings_calls"]
    DatasetRecord,        # text: str, metadata: dict
    EvaluationQuery,      # query: str, answers: list[str], relevant_doc_ids: list[str]
    DataloaderError,
    UnsupportedDatasetError,
    DatasetLoadError,
    DatasetValidationError,
)
```

### Multi-Tenancy / Namespaces

```python
from vectordb.haystack.multi_tenancy.common.types import (
    TenantIsolationStrategy,  # Enum
    TenantIndexResult,        # documents_indexed: int, tenant_id: str, timing_ms: float
    TenantRetrievalResult,    # documents: list, tenant_id: str, timing_ms: float, scores: list
    TenantRAGResult,          # answer: str, retrieval_result: TenantRetrievalResult
    TenantQueryResult,
    TenantStats,
    TenantStatus,             # Enum
)
```

### Evaluation

```python
from vectordb.utils.evaluation import (
    RetrievalMetrics,     # recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate
    QueryResult,          # query, retrieved_ids, retrieved_contents, relevant_ids, scores
    EvaluationResult,     # metrics, query_results, pipeline_name, dataset_name, config
)
```

## 7. Source Walkthrough Map

### Core Exports

| Module | File |
|--------|------|
| **Databases** | `src/vectordb/databases/__init__.py` |
| **Dataloaders** | `src/vectordb/dataloaders/__init__.py` |
| **Utils** | `src/vectordb/utils/__init__.py` |

### Framework Exports

| Module | File |
|--------|------|
| **Haystack** | `src/vectordb/haystack/__init__.py` |
| **LangChain** | `src/vectordb/langchain/__init__.py` |

### Type Definitions

| Module | File |
|--------|------|
| **Dataloaders** | `src/vectordb/dataloaders/types.py` |
| **Multi-Tenancy** | `src/vectordb/haystack/multi_tenancy/common/types.py` |
| **Evaluation** | `src/vectordb/utils/evaluation.py` |

---

**Related Documentation**:

- **Configuration Reference** (`docs/reference/config-reference.md`): All config keys
- **Core VectorDB** (`docs/core/vectordb.md`): Package architecture
