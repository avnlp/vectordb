"""Tests for diversity filtering pipelines in Haystack.

This package contains tests for diversity filtering implementations
in Haystack. Diversity filtering reduces redundancy in search results
by ensuring retrieved documents cover different aspects of the query.

Diversity algorithms:
    - Maximal Marginal Relevance (MMR): Balance relevance and diversity
    - Clustering-based: Group similar documents and select representatives
    - Semantic deduplication: Remove near-duplicate content
    - Coverage-based: Maximize topic coverage in results

Diversity metrics:
    - Intra-list similarity: Average pairwise similarity
    - Coverage: Proportion of query aspects addressed
    - Redundancy ratio: Duplicate information percentage

Database implementations tested:
    - Chroma: Local diversity filtering
    - Milvus: Cluster-based diversity
    - Pinecone: Namespace diversity optimization
    - Qdrant: Payload-based deduplication
    - Weaviate: Cross-reference diversity

Each implementation tests:
    - Diversity metric improvement
    - Relevance preservation
    - Performance with varying lambda values
    - Integration with RAG pipelines
"""
