"""Query enhancement indexing pipelines for LangChain vector database integrations.

Provides unified indexing interfaces across vector databases with consistent
document ingestion for query-enhanced retrieval strategies.

Supported Databases:
    - Qdrant: Cloud-native with payload indexing, optimized for hybrid search
    - Milvus: Enterprise-scale with GPU acceleration, dynamic partitioning
    - Weaviate: GraphQL interface, native hybrid search, modular AI integrations
    - Pinecone: Fully managed serverless with automatic scaling
    - Chroma: Local-first development, zero setup for prototyping

Query Enhancement Strategies:
    Indexing prepares document collections for three enhancement approaches:
    - Multi-Query: Documents embedded once, reused across N query variations
    - HyDE (Hypothetical Document Embeddings): Standard indexing supports
      retrieval by hypothetical answer document embeddings
    - Step-Back: Document metadata and chunking support abstract concept
      retrieval followed by specific detail filtering

Pipeline Features:
    - Automatic collection/index creation with configured parameters
    - Batch document embedding with progress tracking
    - Metadata preservation for filtering during retrieval
    - Idempotent upserts prevent duplicate documents

Example:
    >>> from vectordb.langchain.query_enhancement.indexing import (
    ...     QdrantQueryEnhancementIndexingPipeline,
    ... )
    >>> pipeline = QdrantQueryEnhancementIndexingPipeline("config.yaml")
    >>> pipeline.run()  # Index documents for query-enhanced search
"""

from vectordb.langchain.query_enhancement.indexing.chroma import (
    ChromaQueryEnhancementIndexingPipeline,
)
from vectordb.langchain.query_enhancement.indexing.milvus import (
    MilvusQueryEnhancementIndexingPipeline,
)
from vectordb.langchain.query_enhancement.indexing.pinecone import (
    PineconeQueryEnhancementIndexingPipeline,
)
from vectordb.langchain.query_enhancement.indexing.qdrant import (
    QdrantQueryEnhancementIndexingPipeline,
)
from vectordb.langchain.query_enhancement.indexing.weaviate import (
    WeaviateQueryEnhancementIndexingPipeline,
)


__all__ = [
    "ChromaQueryEnhancementIndexingPipeline",
    "MilvusQueryEnhancementIndexingPipeline",
    "PineconeQueryEnhancementIndexingPipeline",
    "QdrantQueryEnhancementIndexingPipeline",
    "WeaviateQueryEnhancementIndexingPipeline",
]
