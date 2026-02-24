"""Diversity filtering feature for Haystack RAG pipelines.

Diversity filtering addresses the problem of redundant search results by selecting
a diverse subset of documents that cover multiple aspects of a query, rather than
returning many similar documents.

Approaches:
- Maximum Margin Relevance (MMR): Greedy algorithm that balances query relevance
  with diversity by selecting documents that maximize the margin between relevance
  and similarity to already-selected documents.
- Clustering-based: Groups documents into topic clusters using KMeans or HDBSCAN,
  then selects representative documents from each cluster to ensure coverage.
- Embedding-distance: Uses pairwise cosine distances in embedding space to measure
  and maximize diversity among selected documents.

Use cases:
- Research queries: Ensures comprehensive coverage of different perspectives on a topic
- E-commerce: Shows varied product options rather than many similar items
- News retrieval: Provides balanced viewpoints from different sources
- Academic search: Surfaces papers from different research clusters

Supports all 5 vector databases:
- Qdrant, Pinecone, Weaviate, Chroma, Milvus

Example usage:
    from vectordb.haystack.diversity_filtering.pipelines import qdrant_indexing

    result = qdrant_indexing.run_indexing("configs/qdrant/triviaqa/indexing.yaml")

    from vectordb.haystack.diversity_filtering.pipelines.chroma_search import (
        ChromaDiversitySearchPipeline,
    )

    # Initialize pipeline once (components loaded once, reused across searches)
    pipeline = ChromaDiversitySearchPipeline(
        "configs/chroma/triviaqa/search.yaml"
    )

    # Execute multiple searches efficiently
    result1 = pipeline.search("What is AI?")
    result2 = pipeline.search("How does machine learning work?")
"""

from vectordb.haystack.diversity_filtering import pipelines, utils


__all__ = ["pipelines", "utils"]
