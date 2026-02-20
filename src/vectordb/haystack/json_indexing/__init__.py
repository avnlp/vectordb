"""JSON indexing and search pipelines for vector databases.

Provides simple, configuration-driven indexing and search interfaces for:
- Milvus
- Pinecone
- Qdrant
- Weaviate
- Chroma

All pipelines support JSON metadata filtering and are fully programmatic
(no CLI or interactive REPL).

Example:
    Index documents:
        from vectordb.haystack.json_indexing.indexing import MilvusJSONIndexer

        indexer = MilvusJSONIndexer("configs/milvus/triviaqa.yaml")
        result = indexer.run()

    Search with filters:
        from vectordb.haystack.json_indexing.search import MilvusJSONSearcher

        searcher = MilvusJSONSearcher("configs/milvus/triviaqa.yaml")
        results = searcher.search(
            query="What is machine learning?",
            filters={"topic": "AI"},
            top_k=10
        )
"""

from vectordb.haystack.json_indexing.indexing import (
    ChromaJSONIndexer,
    MilvusJSONIndexer,
    PineconeJSONIndexer,
    QdrantJSONIndexer,
    WeaviateJSONIndexer,
)
from vectordb.haystack.json_indexing.search import (
    ChromaJSONSearcher,
    MilvusJSONSearcher,
    PineconeJSONSearcher,
    QdrantJSONSearcher,
    WeaviateJSONSearcher,
)


__all__ = [
    # Indexing
    "MilvusJSONIndexer",
    "PineconeJSONIndexer",
    "QdrantJSONIndexer",
    "WeaviateJSONIndexer",
    "ChromaJSONIndexer",
    # Search
    "MilvusJSONSearcher",
    "PineconeJSONSearcher",
    "QdrantJSONSearcher",
    "WeaviateJSONSearcher",
    "ChromaJSONSearcher",
]
