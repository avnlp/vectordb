"""Tests for parent document retrieval pipelines (LangChain).

This package contains tests for parent document retrieval implementations
in LangChain. This technique improves context quality by retrieving larger
parent documents while using smaller chunks for embedding-based search.

Parent document retrieval workflow:
    1. Split documents into small chunks for embedding
    2. Index chunks with parent document references
    3. Search using chunk embeddings
    4. Return parent documents for context

Benefits:
    - Better embedding quality (focused chunks)
    - Richer context (full parent documents)
    - Reduced token usage (no chunk overlap)

Database implementations tested:
    - Chroma: Parent-child relationship storage
    - Milvus: Partition-based parent storage
    - Pinecone: Metadata-based parent linking
    - Qdrant: Payload-based parent references
    - Weaviate: Cross-reference parent documents

Each implementation tests:
    - Document chunking strategies
    - Parent-child relationship mapping
    - Retrieval accuracy
    - Context completeness
    - Integration with LangChain chains
"""
