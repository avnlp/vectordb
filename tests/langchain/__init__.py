"""LangChain integration tests.

This package contains comprehensive tests for all LangChain integrations
with vector databases. Tests cover:

Database integrations:
    - Chroma: Local persistent vector database
    - Milvus: Cloud-native vector database
    - Pinecone: Managed vector database service
    - Qdrant: High-performance vector search engine
    - Weaviate: Graph-vector database

LangChain components tested:
    - VectorStores: Database wrapper implementations
    - Retrievers: Search and retrieval components
    - Agentic RAG: Agent-based RAG pipeline components
    - Cost-Optimized RAG: Cost-aware retrieval strategies
    - MMR: Maximal Marginal Relevance for diversity
    - Metadata Filtering: Attribute-based filtering
    - Multi-Tenancy: Tenant isolation features
    - Namespaces: Collection partitioning
    - Parent Document Retrieval: Hierarchical retrieval
    - Query Enhancement: Query transformation and expansion
    - Semantic Search: Embedding-based similarity search
    - Sparse Indexing: Hybrid search capabilities
    - Contextual Compression: Context-aware document compression
    - Diversity Filtering: Result diversification algorithms
    - Hybrid Indexing: Combined dense/sparse retrieval

Each integration is tested for:
    - Indexing: Document storage with embeddings
    - Search: Similarity search with filters
    - Chain integration: LangChain Expression Language (LCEL)
    - Retriever compliance: LangChain Retriever interface
"""
