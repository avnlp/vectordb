"""Hybrid indexing pipelines for vector databases.

This module provides unified indexing pipelines for hybrid (dense + sparse)
vector search across multiple database backends. Hybrid search combines the
strengths of semantic similarity (dense embeddings) and lexical matching
(sparse embeddings) for improved retrieval accuracy.

Hybrid Search Overview:
    Hybrid search combines two retrieval approaches:

    1. Dense Embeddings: Capture semantic meaning using neural networks
       (e.g., sentence-transformers, OpenAI embeddings). Good for conceptual
       similarity but may miss exact keyword matches.

    2. Sparse Embeddings: Capture lexical/keyword matches using approaches
       like SPLADE (Sparse Lexical and Expansion Model) or BM25. Good for
       exact term matching but lack semantic understanding.

    The two result sets are fused using ranking algorithms like RRF
    (Reciprocal Rank Fusion) or linear score combination to produce
    the final ranked results.

Sparse Embedding Approaches:
    - SPLADE: Learns to expand queries/documents with relevant terms using
      a pretrained language model, producing learned sparse representations
    - BM25: Traditional TF-IDF based scoring (used natively by Weaviate)
    - Learned sparse models: Use neural networks to predict term importance

Fusion Strategies:
    - RRF (Reciprocal Rank Fusion): Combines rankings from multiple sources
      using the formula: score = sum(1.0 / (k + rank)) for each result.
      K is typically 60. RRF is robust and doesn't require score calibration.
    - Linear Combination: Weighted sum of normalized scores from each
      retrieval method: final_score = alpha * dense_score + (1-alpha) * sparse_score

Supported Backends:
    - Milvus: Native hybrid search with sparse vector support and RRF
    - Pinecone: Native sparse vector support with alpha weighting
    - Qdrant: Native hybrid search with sparse vectors
    - Weaviate: Native BM25 + vector hybrid (no sparse embedder needed)
    - Chroma: Manual fusion of dense and sparse search results

Example:
    >>> from vectordb.haystack.hybrid_indexing.indexing import (
    ...     MilvusHybridIndexingPipeline,
    ... )
    >>> indexer = MilvusHybridIndexingPipeline(
    ...     config_path="configs/milvus/triviaqa.yaml"
    ... )
    >>> result = indexer.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
"""

from vectordb.haystack.hybrid_indexing.indexing.chroma import (
    ChromaHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.indexing.milvus import (
    MilvusHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.indexing.pinecone import (
    PineconeHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.indexing.qdrant import (
    QdrantHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.indexing.weaviate import (
    WeaviateHybridIndexingPipeline,
)


__all__ = [
    "ChromaHybridIndexingPipeline",
    "MilvusHybridIndexingPipeline",
    "PineconeHybridIndexingPipeline",
    "QdrantHybridIndexingPipeline",
    "WeaviateHybridIndexingPipeline",
]
