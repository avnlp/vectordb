"""Tests for parent document retrieval in Haystack.

This package contains tests for parent document retrieval implementations
in Haystack. Parent document retrieval returns complete parent documents
when child chunks match a query, providing better context for generation.

Parent document retrieval workflow:
    1. Index full documents as parents
    2. Create and index smaller child chunks with parent references
    3. Search child chunks for query matches
    4. Return parent documents of matching children

Benefits over standard chunking:
    - Preserves complete document context
    - Avoids chunk boundary issues
    - Maintains document coherence
    - Better for long-form content

Implementation approaches:
    - Two-phase indexing: Parents and children separately
    - Metadata linking: Child chunks reference parent IDs
    - Hierarchical retrieval: Navigate from children to parents

Database implementations tested:
    - Chroma: Parent-child relationship storage
    - Milvus: Entity-based parent retrieval
    - Pinecone: Metadata-based parent linking
    - Qdrant: Payload-based parent references
    - Weaviate: Cross-reference parent documents

Each implementation tests:
    - Parent-child relationship integrity
    - Retrieval accuracy
    - Performance with large parent documents
    - Edge cases (orphaned children, missing parents)
"""
