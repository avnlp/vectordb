"""Query-enhanced search pipelines for LangChain vector database integrations.

Provides unified search interfaces across vector databases with integrated
query transformation strategies for improved retrieval quality.

Supported Databases:
    - Qdrant: Hybrid search with HNSW indexing, ideal for multi-query fusion
    - Milvus: GPU-accelerated ANN search with dynamic partition pruning
    - Weaviate: Native hybrid search combining sparse BM25 and dense vectors
    - Pinecone: Metadata-filtered vector search with automatic index management
    - Chroma: Local vector search with persistent embedding storage

Query Enhancement Strategies:
    - Multi-Query: Expands single query into N semantic variations using LLM.
      Each variation retrieves independently; results fused via RRF algorithm.
    - HyDE (Hypothetical Document Embeddings): LLM generates M hypothetical
      answer documents for the query. These are embedded and used as search
      vectors, matching documents to potential answers rather than queries.
    - Step-Back: Abstracts specific factual queries into higher-level conceptual
      questions. Retrieves broad context documents first, then filters to
      specific details for comprehensive yet precise answers.

Fusion and Ranking:
    - RRF (Reciprocal Rank Fusion): Combines ranked lists using score = sum(1/(k+rank))
    - Weighted fusion: Configurable score weighting for different result sources
    - Deduplication: Content-based deduplication prevents duplicate documents

Example:
    >>> from vectordb.langchain.query_enhancement.search import (
    ...     QdrantQueryEnhancementSearchPipeline,
    ... )
    >>> searcher = QdrantQueryEnhancementSearchPipeline("config.yaml")
    >>> results = searcher.run("What are transformer architectures?")
"""

from vectordb.langchain.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)
from vectordb.langchain.query_enhancement.search.chroma import (
    ChromaQueryEnhancementSearchPipeline,
)
from vectordb.langchain.query_enhancement.search.milvus import (
    MilvusQueryEnhancementSearchPipeline,
)
from vectordb.langchain.query_enhancement.search.pinecone import (
    PineconeQueryEnhancementSearchPipeline,
)
from vectordb.langchain.query_enhancement.search.qdrant import (
    QdrantQueryEnhancementSearchPipeline,
)
from vectordb.langchain.query_enhancement.search.weaviate import (
    WeaviateQueryEnhancementSearchPipeline,
)


__all__ = [
    "BaseQueryEnhancementSearchPipeline",
    "ChromaQueryEnhancementSearchPipeline",
    "MilvusQueryEnhancementSearchPipeline",
    "PineconeQueryEnhancementSearchPipeline",
    "QdrantQueryEnhancementSearchPipeline",
    "WeaviateQueryEnhancementSearchPipeline",
]
