"""Semantic search pipelines for vector databases.

This module provides end-to-end semantic search pipelines that combine
document indexing with vector-based similarity search. The pipelines
leverage embedding models to convert text into dense vectors, enabling
semantically-aware document retrieval.

Key Features:
    - Dense vector indexing using various embedding models
    - Similarity search with configurable metrics (cosine, euclidean, dot product)
    - Optional RAG answer generation
    - Metadata filtering support
    - Result diversification for improved recall

Architecture:
    Indexing: Document -> Embedding Model -> Vector Database
    Search: Query -> Embedding Model -> Vector Search -> Results

Supported Vector Databases:
    - Pinecone: Cloud-native vector database with managed infrastructure
    - Chroma: Embedded vector database for local development
    - Milvus: Open-source vector database with high scalability
    - Qdrant: High-performance vector search with rich metadata filtering
    - Weaviate: Schema-based vector search with GraphQL interface

Comparison with LangChain Integration:

    Haystack Integration (this module):
        - Pipeline-based architecture with explicit component connections
        - Native Haystack Document format throughout
        - Built-in RAG prompt templates and generators
        - Easier to customize with Haystack's component ecosystem
        - Better for production RAG pipelines with complex preprocessing

    LangChain Integration (vectordb.langchain):
        - Chain-based composition with LCEL (LangChain Expression Language)
        - LangChain Document format with different metadata conventions
        - More flexible but requires more boilerplate for standard RAG
        - Better integration with LangChain's agent and tool ecosystem
        - Better for rapid prototyping and agent-based applications

    Both implementations share the same underlying VectorDB classes
    (PineconeVectorDB, ChromaVectorDB, etc.), ensuring consistent
database behavior across frameworks. The choice between Haystack and
LangChain depends on your use case:

    Choose Haystack when:
        - Building production RAG pipelines
        - Need explicit control over preprocessing and postprocessing
        - Want built-in evaluation and optimization tools
        - Prefer pipeline visualization and debugging

    Choose LangChain when:
        - Building agent-based applications
        - Need rapid prototyping with flexible composition
        - Want integration with LangChain's tool ecosystem
        - Prefer chain-based programming model

Usage:
    Indexing:
        >>> from vectordb.haystack.semantic_search import (
        ...     PineconeSemanticIndexingPipeline,
        ... )
        >>> indexer = PineconeSemanticIndexingPipeline("config.yaml")
        >>> stats = indexer.run()
        >>> print(f"Indexed {stats['documents_indexed']} documents")

    Search:
        >>> from vectordb.haystack.semantic_search import PineconeSemanticSearchPipeline
        >>> searcher = PineconeSemanticSearchPipeline("config.yaml")
        >>> results = searcher.search("machine learning applications", top_k=10)
        >>> for doc in results["documents"]:
        ...     print(f"- {doc.content[:100]}...")

    With RAG:
        >>> results = searcher.search("What is RAG?", top_k=5)
        >>> if "answer" in results:
        ...     print(f"Answer: {results['answer']}")
"""

# Indexing pipelines
from vectordb.haystack.semantic_search.indexing import (
    ChromaSemanticIndexingPipeline,
    MilvusSemanticIndexingPipeline,
    PineconeSemanticIndexingPipeline,
    QdrantSemanticIndexingPipeline,
    WeaviateSemanticIndexingPipeline,
)

# Search pipelines
from vectordb.haystack.semantic_search.search import (
    ChromaSemanticSearchPipeline,
    MilvusSemanticSearchPipeline,
    PineconeSemanticSearchPipeline,
    QdrantSemanticSearchPipeline,
    WeaviateSemanticSearchPipeline,
)


__all__ = [
    # Indexing
    "MilvusSemanticIndexingPipeline",
    "QdrantSemanticIndexingPipeline",
    "PineconeSemanticIndexingPipeline",
    "WeaviateSemanticIndexingPipeline",
    "ChromaSemanticIndexingPipeline",
    # Search
    "MilvusSemanticSearchPipeline",
    "QdrantSemanticSearchPipeline",
    "PineconeSemanticSearchPipeline",
    "WeaviateSemanticSearchPipeline",
    "ChromaSemanticSearchPipeline",
]
