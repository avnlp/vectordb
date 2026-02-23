"""Contextual compression pipelines for vector database retrieval.

Contextual compression is a technique to filter out irrelevant chunks from retrieved
documents, keeping only the most relevant context for Large Language Models (LLMs).
This reduces noise and token consumption in Retrieval-Augmented Generation (RAG).

Compression Approaches:
    1. LLM-based compression: Uses an LLM (e.g., GPT-4o-mini) to extract only the
       relevant portions of documents based on the query context.
    2. Embedding-based compression: Uses reranking models (e.g., cross-encoders,
       Cohere, Voyage) to score and filter documents by relevance.

Benefits for RAG:
    - Reduces token count in LLM context windows, lowering costs and latency.
    - Improves answer quality by removing distracting, irrelevant information.
    - Enables retrieving more documents initially, then compressing to fit limits.

Supported Databases:
    - Qdrant, Weaviate, Milvus, Pinecone, Chroma

Example Usage:
    >>> from vectordb.haystack.contextual_compression import QdrantCompressionSearch
    >>> pipeline = QdrantCompressionSearch("configs/qdrant/reranking.yaml")
    >>> results = pipeline.run("What is RAG?", top_k=5)
"""

from vectordb.haystack.contextual_compression.base import (
    BaseContextualCompressionPipeline,
)
from vectordb.haystack.contextual_compression.compression_utils import (
    CompressorFactory,
    RankerResult,
    TokenCounter,
    format_compression_results,
    prepare_retrieval_batch,
)
from vectordb.haystack.contextual_compression.search import (
    ChromaCompressionSearch,
    MilvusCompressionSearch,
    PineconeCompressionSearch,
    QdrantCompressionSearch,
    WeaviateCompressionSearch,
)


__all__ = [
    "BaseContextualCompressionPipeline",
    "QdrantCompressionSearch",
    "WeaviateCompressionSearch",
    "MilvusCompressionSearch",
    "PineconeCompressionSearch",
    "ChromaCompressionSearch",
    "RankerResult",
    "CompressorFactory",
    "TokenCounter",
    "prepare_retrieval_batch",
    "format_compression_results",
]
