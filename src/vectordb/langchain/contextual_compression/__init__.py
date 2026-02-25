"""Contextual compression implementations for vector databases.

This module provides contextual compression pipelines for RAG that filter
or summarize retrieved documents before generation. Contextual compression
improves retrieval quality by reducing noise and focusing on relevant content.

Contextual Compression Techniques:
    - Reranking: Uses a cross-encoder or LLM to reorder retrieved documents
      based on their relevance to the specific query, improving precision
      over simple vector similarity scores.

    - LLM Extraction: Uses a language model to extract only the most relevant
      passages from retrieved documents, compressing long documents into
      concise, query-focused snippets.

Benefits:
    - Reduced context window usage for LLM generation
    - Higher signal-to-noise ratio in retrieved content
    - Better handling of long documents with mixed relevance
    - Improved answer quality through focused context

Database Support:
    All supported vector databases (Pinecone, Weaviate, Chroma, Milvus, Qdrant)
    can use contextual compression through the retrieval layer before
    generation.

Key Classes:
    Indexing Pipelines:
        - ChromaContextualCompressionIndexingPipeline
        - MilvusContextualCompressionIndexingPipeline
        - PineconeContextualCompressionIndexingPipeline
        - QdrantContextualCompressionIndexingPipeline
        - WeaviateContextualCompressionIndexingPipeline

    Search Pipelines:
        - ChromaContextualCompressionSearchPipeline
        - MilvusContextualCompressionSearchPipeline
        - PineconeContextualCompressionSearchPipeline
        - QdrantContextualCompressionSearchPipeline
        - WeaviateContextualCompressionSearchPipeline

Example:
    >>> from vectordb.langchain.contextual_compression import (
    ...     PineconeContextualCompressionSearchPipeline,
    ... )
    >>> pipeline = PineconeContextualCompressionSearchPipeline("config.yaml")
    >>> result = pipeline.search("What is machine learning?", top_k=5)
    >>> for doc in result["documents"]:
    ...     if "rerank_score" in doc.metadata:
    ...         print(f"Relevance: {doc.metadata['rerank_score']:.2f}")
    ...     print(f"Content: {doc.page_content[:200]}...")
"""

from vectordb.langchain.contextual_compression.indexing import (
    ChromaContextualCompressionIndexingPipeline,
    MilvusContextualCompressionIndexingPipeline,
    PineconeContextualCompressionIndexingPipeline,
    QdrantContextualCompressionIndexingPipeline,
    WeaviateContextualCompressionIndexingPipeline,
)
from vectordb.langchain.contextual_compression.search import (
    ChromaContextualCompressionSearchPipeline,
    MilvusContextualCompressionSearchPipeline,
    PineconeContextualCompressionSearchPipeline,
    QdrantContextualCompressionSearchPipeline,
    WeaviateContextualCompressionSearchPipeline,
)


__all__ = [
    "ChromaContextualCompressionIndexingPipeline",
    "MilvusContextualCompressionIndexingPipeline",
    "PineconeContextualCompressionIndexingPipeline",
    "QdrantContextualCompressionIndexingPipeline",
    "WeaviateContextualCompressionIndexingPipeline",
    "ChromaContextualCompressionSearchPipeline",
    "MilvusContextualCompressionSearchPipeline",
    "PineconeContextualCompressionSearchPipeline",
    "QdrantContextualCompressionSearchPipeline",
    "WeaviateContextualCompressionSearchPipeline",
]
