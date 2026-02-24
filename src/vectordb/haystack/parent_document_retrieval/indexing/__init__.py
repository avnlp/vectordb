"""Parent document retrieval indexing pipelines.

This module provides indexing pipelines that build hierarchical document structures
for parent document retrieval. The key concept is the chunk-to-parent mapping:

Parent Document Retrieval Strategy:
    1. Documents are split hierarchically into parent and child chunks
    2. Child chunks (leaves) are embedded and indexed in the vector database
    3. Parent chunks are stored in a Haystack DocumentStore
    4. Child metadata contains parent_id to link back to parent documents

Benefits of Parent Document Retrieval:
    - Context Preservation: Parents provide broader context than small chunks
    - Better Coverage: Multi-chunk concepts are retrieved as complete parent docs
    - Token Efficiency: Returning one parent is more efficient than many children
    - Reduced Redundancy: Parents contain merged information from children

Hierarchical Chunking Strategies:
    - Parent Size: Larger chunks (e.g., 100 words) containing multiple concepts
    - Child Size: Smaller chunks (e.g., 25 words) for precise matching
    - Overlap: Small overlap between chunks prevents context loss at boundaries

Pipeline Flow:
    1. Load documents from configured dataset
    2. Apply HierarchicalDocumentSplitter to create parent/child relationships
    3. Store parent documents in InMemoryDocumentStore
    4. Embed child documents and index in vector database
    5. Return statistics about indexed documents

Available Pipelines:
    - ChromaParentDocIndexingPipeline: For Chroma vector database
    - MilvusParentDocIndexingPipeline: For Milvus vector database
    - PineconeParentDocIndexingPipeline: For Pinecone vector database
    - QdrantParentDocIndexingPipeline: For Qdrant vector database
    - WeaviateParentDocIndexingPipeline: For Weaviate vector database

Usage:
    >>> from vectordb.haystack.parent_document_retrieval import (
    ...     ChromaParentDocIndexingPipeline,
    ... )
    >>> pipeline = ChromaParentDocIndexingPipeline("config.yaml")
    >>> stats = pipeline.run(limit=100)
    >>> print(f"Indexed {stats['num_parents']} parents, {stats['num_leaves']} leaves")
"""
