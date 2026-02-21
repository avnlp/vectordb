"""Tests for semantic search pipelines (LangChain).

This package contains tests for semantic (embedding-based) search implementations
in LangChain. Semantic search retrieves documents based on vector similarity
between query and document embeddings.

Semantic search process:
    1. Encode query into dense embedding vector
    2. Compute similarity with document embeddings
    3. Return top-k most similar documents

Similarity metrics tested:
    - Cosine similarity: Measures angle between vectors (most common)
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
    - Integration with LangChain retrievers and LCEL
"""
