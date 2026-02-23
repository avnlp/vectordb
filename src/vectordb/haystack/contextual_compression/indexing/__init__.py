"""Contextual compression indexing pipelines for all vector databases.

Indexing pipelines prepare document stores for contextual compression search.
These pipelines are separate from the compression logic and focus on:
    - Loading datasets (TriviaQA, ARC, PopQA, FactScore, EarningsCall)
    - Generating dense embeddings (Qwen3, MiniLM, etc.)
    - Storing vectors in database-specific collections

Indexing Workflow:
    1. Load dataset via DatasetRegistry
    2. Initialize embedder (SentenceTransformersTextEmbedder)
    3. Generate embeddings in batches
    4. Store in vector database with metadata

Each pipeline is database-specific (Milvus, Pinecone, Qdrant, Chroma, Weaviate)
and works independently of json_indexing pipelines.
"""

from vectordb.haystack.contextual_compression.indexing.base_indexing import (
    BaseIndexingPipeline,
)
from vectordb.haystack.contextual_compression.indexing.chroma_indexing import (
    ChromaIndexingPipeline,
)
from vectordb.haystack.contextual_compression.indexing.milvus_indexing import (
    MilvusIndexingPipeline,
)
from vectordb.haystack.contextual_compression.indexing.pinecone_indexing import (
    PineconeIndexingPipeline,
)
from vectordb.haystack.contextual_compression.indexing.qdrant_indexing import (
    QdrantIndexingPipeline,
)
from vectordb.haystack.contextual_compression.indexing.weaviate_indexing import (
    WeaviateIndexingPipeline,
)


__all__ = [
    "BaseIndexingPipeline",
    "MilvusIndexingPipeline",
    "PineconeIndexingPipeline",
    "QdrantIndexingPipeline",
    "ChromaIndexingPipeline",
    "WeaviateIndexingPipeline",
]
