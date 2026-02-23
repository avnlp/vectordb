"""Search pipelines with contextual compression for all vector databases.

Provides retrieval + compression pipelines that fetch documents from vector stores
and apply contextual compression before returning results.

Search Pipeline Flow:
    1. Dense Retrieval: Query embedding -> vector search -> fetch top_k*2 documents
    2. Compression: Apply reranker or LLM extractor to filter documents
    3. Return: Top-k most relevant documents after compression

Supported Compression Types:
    - Reranking: Cross-encoder, Cohere, Voyage, BGE models
    - LLM Extraction: OpenAI-compatible APIs (GPT-4o-mini, Groq, etc.)

Prerequisites:
    Documents must be indexed using the corresponding indexing pipeline
    (e.g., QdrantIndexingPipeline for QdrantCompressionSearch).

Example:
    >>> from vectordb.haystack.contextual_compression.search import (
    ...     QdrantCompressionSearch,
    ... )
    >>> search = QdrantCompressionSearch("configs/qdrant/triviaqa/reranking.yaml")
    >>> results = search.run("What is machine learning?", top_k=5)
"""

from vectordb.haystack.contextual_compression.search.chroma_search import (
    ChromaCompressionSearch,
)
from vectordb.haystack.contextual_compression.search.milvus_search import (
    MilvusCompressionSearch,
)
from vectordb.haystack.contextual_compression.search.pinecone_search import (
    PineconeCompressionSearch,
)
from vectordb.haystack.contextual_compression.search.qdrant_search import (
    QdrantCompressionSearch,
)
from vectordb.haystack.contextual_compression.search.weaviate_search import (
    WeaviateCompressionSearch,
)


__all__ = [
    "MilvusCompressionSearch",
    "PineconeCompressionSearch",
    "QdrantCompressionSearch",
    "ChromaCompressionSearch",
    "WeaviateCompressionSearch",
]
