"""Contextual compression search pipelines for vector databases.

This module provides search pipelines that apply contextual compression
techniques to retrieved documents before returning results. These pipelines
improve retrieval quality by filtering noise and focusing on query-relevant
content.

Search Pipeline Process:
    1. Embed query using configured embedder
    2. Retrieve top-k documents from vector database
    3. Apply contextual compression (reranking, extraction)
    4. Return compressed/filtered documents

Compression Techniques Applied:
    - Reranking: Reorder documents by relevance using cross-encoders
    - LLM Extraction: Extract relevant passages from long documents
    - Filtering: Remove low-relevance documents based on score threshold

Available Search Pipelines:
    - ChromaContextualCompressionSearchPipeline: Chroma with compression
    - MilvusContextualCompressionSearchPipeline: Milvus with compression
    - PineconeContextualCompressionSearchPipeline: Pinecone with compression
    - QdrantContextualCompressionSearchPipeline: Qdrant with compression
    - WeaviateContextualCompressionSearchPipeline: Weaviate with compression

Configuration Options:
    - reranker: Model to use for reranking (cross-encoder, LLM-based)
    - compression_ratio: Target size reduction for extraction
    - min_score_threshold: Minimum relevance score to keep document
    - top_k_initial: Number of documents to retrieve before compression
    - top_k_final: Number of documents to return after compression
"""

from vectordb.langchain.contextual_compression.search.chroma import (
    ChromaContextualCompressionSearchPipeline,
)
from vectordb.langchain.contextual_compression.search.milvus import (
    MilvusContextualCompressionSearchPipeline,
)
from vectordb.langchain.contextual_compression.search.pinecone import (
    PineconeContextualCompressionSearchPipeline,
)
from vectordb.langchain.contextual_compression.search.qdrant import (
    QdrantContextualCompressionSearchPipeline,
)
from vectordb.langchain.contextual_compression.search.weaviate import (
    WeaviateContextualCompressionSearchPipeline,
)


__all__ = [
    "ChromaContextualCompressionSearchPipeline",
    "MilvusContextualCompressionSearchPipeline",
    "PineconeContextualCompressionSearchPipeline",
    "QdrantContextualCompressionSearchPipeline",
    "WeaviateContextualCompressionSearchPipeline",
]
