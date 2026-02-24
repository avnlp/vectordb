"""MMR search pipelines for vector databases.

This module provides Maximal Marginal Relevance (MMR) search pipelines that
retrieve diverse, relevant documents by balancing similarity with redundancy
reduction. MMR search is ideal for applications requiring multiple perspectives
or avoiding repetitive results.

The MMR Algorithm:
    1. Fetch a larger candidate set (mmr_k) using standard similarity search
    2. Iteratively select documents that maximize: relevance - λ * redundancy
    3. Return the top_k most diverse, relevant results

Database-Specific Search Classes:
    - ChromaMMRSearchPipeline: Local vector database with MMR support for
        development and testing environments
    - MilvusMMRSearchPipeline: Scalable open-source vector database with
        partition-aware MMR retrieval
    - PineconeMMRSearchPipeline: Managed cloud service with optimized MMR
        candidate fetching and metadata filtering
    - QdrantMMRSearchPipeline: High-performance search with payload-based
        pre-filtering before MMR diversification
    - WeaviateMMRSearchPipeline: GraphQL-enabled vector search with MMR
        integration for knowledge graph applications

Each search pipeline supports:
    - Configurable lambda parameter for relevance-diversity trade-off
    - Candidate pre-fetching (mmr_k) before MMR selection
    - Top-k result limiting after diversity filtering
    - Query-time metadata filtering combined with MMR
    - Async search operations for production workloads
"""

from vectordb.langchain.mmr.search.chroma import ChromaMMRSearchPipeline
from vectordb.langchain.mmr.search.milvus import MilvusMMRSearchPipeline
from vectordb.langchain.mmr.search.pinecone import PineconeMMRSearchPipeline
from vectordb.langchain.mmr.search.qdrant import QdrantMMRSearchPipeline
from vectordb.langchain.mmr.search.weaviate import WeaviateMMRSearchPipeline


__all__ = [
    "ChromaMMRSearchPipeline",
    "MilvusMMRSearchPipeline",
    "PineconeMMRSearchPipeline",
    "QdrantMMRSearchPipeline",
    "WeaviateMMRSearchPipeline",
]
