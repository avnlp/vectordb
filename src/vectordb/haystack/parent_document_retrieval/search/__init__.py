"""Search pipelines for parent document retrieval.

This module provides search pipelines that implement parent document retrieval,
a strategy where child chunks are indexed for precise matching while parent
documents are returned for comprehensive context.

Parent Document Retrieval Strategy:
    1. Index both parent documents (large) and child chunks (small)
    2. Embed and search child chunks for precise semantic matching
    3. Use AutoMergingRetriever to resolve child matches to parent documents
    4. Return parent documents with broader context for LLM prompts

The key insight: small child chunks enable precise semantic matching during
search, while parent documents provide the comprehensive context needed for
generative tasks.

Architecture:
    Each database implementation follows this search flow:
    1. Embed query using the same model as indexing
    2. Query vector database for child chunk matches (oversample for recall)
    3. Use Haystack's AutoMergingRetriever to group children by parent
    4. Return parent documents when sufficient children match

Auto-Merging Logic:
    The AutoMergingRetriever groups matched children by their parent_id and
    returns the parent document when enough children match (merge_threshold).
    This balances precision (children) with context (parents).

Key Components:
    - PineconeParentDocSearchPipeline: Pinecone parent document search
    - MilvusParentDocSearchPipeline: Milvus parent document search
    - QdrantParentDocSearchPipeline: Qdrant parent document search
    - WeaviateParentDocSearchPipeline: Weaviate parent document search
    - ChromaParentDocSearchPipeline: Chroma parent document search

Usage:
    >>> from vectordb.haystack.parent_document_retrieval import (
    ...     PineconeParentDocIndexingPipeline,
    ...     PineconeParentDocSearchPipeline,
    ... )
    >>> # Index phase
    >>> indexer = PineconeParentDocIndexingPipeline("config.yaml")
    >>> stats = indexer.run(limit=100)
    >>> # Search phase
    >>> search = PineconeParentDocSearchPipeline(
    ...     "config.yaml", parent_store=indexer.parent_store
    ... )
    >>> results = search.search("What is machine learning?", top_k=5)
    >>> for doc in results["documents"]:
    ...     print(doc.content[:200])

Integration Points:
    - vectordb.haystack.parent_document_retrieval.indexing: Complementary indexing
    - vectordb.haystack.parent_document_retrieval.utils: ID and metadata utilities
    - haystack.components.retrievers.auto_merging_retriever: AutoMergingRetriever
"""
