"""Parent Document Retrieval pipelines using Haystack components.

This module provides simplified parent document retrieval using Haystack's
HierarchicalDocumentSplitter for creating document hierarchies and
AutoMergingRetriever for retrieving parent documents.

Parent Document Retrieval Strategy:
    1. Split documents into smaller "child" chunks (leaf nodes)
    2. Group related children into "parent" chunks (parent nodes)
    3. Index child chunks in the vector database for retrieval
    4. Store parent documents in a DocumentStore for final retrieval
    5. When children match, return the parent document instead

Benefits:
    - Larger context in retrieved results (parent contains multiple children)
    - Better coverage of concepts spanning multiple chunks
    - Reduced token usage compared to returning many small chunks

Limitations:
    - Requires additional storage for parent documents
    - Parent/child relationship must be maintained in metadata
    - May retrieve more context than needed for simple queries

Usage:
    >>> from vectordb.haystack.parent_document_retrieval import (
    ...     ChromaParentDocIndexingPipeline,
    ... )
    >>> pipeline = ChromaParentDocIndexingPipeline("config.yaml")
    >>> stats = pipeline.run(limit=100)
"""

# Import indexing pipelines
from .indexing.chroma import ChromaParentDocIndexingPipeline
from .indexing.milvus import MilvusParentDocIndexingPipeline
from .indexing.pinecone import PineconeParentDocIndexingPipeline
from .indexing.qdrant import QdrantParentDocIndexingPipeline
from .indexing.weaviate import WeaviateParentDocIndexingPipeline
from .search.chroma import ChromaParentDocSearchPipeline

# Import search pipelines
from .search.milvus import MilvusParentDocSearchPipeline
from .search.pinecone import PineconeParentDocSearchPipeline
from .search.qdrant import QdrantParentDocSearchPipeline
from .search.weaviate import WeaviateParentDocSearchPipeline


__all__ = [
    # Indexing pipelines
    "MilvusParentDocIndexingPipeline",
    "PineconeParentDocIndexingPipeline",
    "QdrantParentDocIndexingPipeline",
    "WeaviateParentDocIndexingPipeline",
    "ChromaParentDocIndexingPipeline",
    # Search pipelines
    "MilvusParentDocSearchPipeline",
    "PineconeParentDocSearchPipeline",
    "QdrantParentDocSearchPipeline",
    "WeaviateParentDocSearchPipeline",
    "ChromaParentDocSearchPipeline",
]
