"""Tests for semantic search pipelines.

This package contains tests for semantic (embedding-based) search pipelines
that retrieve documents based on vector similarity. These pipelines form the
foundation of RAG (Retrieval-Augmented Generation) systems.

Semantic search process:
    1. Encode query into embedding vector
    2. Compute similarity with document embeddings
    3. Return top-k most similar documents

Similarity metrics tested:
    - Cosine similarity: Measures angle between vectors
    - Dot product: Measures vector alignment
    - Euclidean distance: Measures vector separation

Database implementations tested:
    - Chroma: Local persistent semantic search
    - Milvus: Cloud-native high-performance search
    - Pinecone: Managed service with metadata filtering
    - Qdrant: On-premise and cloud semantic search
    - Weaviate: Graph-vector hybrid search

Each implementation tests:
    - Document indexing with embeddings
    - Query encoding and similarity computation
    - Top-k retrieval with various metrics
    - Metadata filtering during search
    - Pipeline integration with Haystack
"""
