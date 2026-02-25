"""Multi-tenancy indexing pipelines for LangChain vector databases.

This module provides document indexing pipelines that prepare vector stores
for multi-tenant usage. Each pipeline handles document loading, embedding
generation, and tenant-isolated indexing across all supported databases.

Multi-Tenancy Indexing Architecture:
    All multi-tenancy indexing pipelines follow a consistent pattern:

    1. Document Loading: Load documents from configured data sources
       (TriviaQA, ARC, PopQA, FactScore, EarningsCall datasets)

    2. Embedding Generation: Generate dense vector embeddings using
       configured embedding models (OpenAI, HuggingFace, etc.)

    3. Tenant-Isolated Indexing: Index documents into tenant-specific
       namespaces, collections, or partitions to ensure data isolation

Tenant Isolation Mechanisms:
    Different databases implement tenant isolation differently:
    - Pinecone: Namespace-based isolation within a single index
    - Weaviate: Collection-based isolation (each tenant = one collection)
    - Chroma: Collection-based isolation with separate collections
    - Milvus: Partition-based isolation within a collection
    - Qdrant: Collection-based isolation with dedicated collections

Pipeline Consistency:
    All pipelines share identical interfaces:
    - __init__(config_or_path): Initialize from dict or YAML file path
    - run() -> dict: Execute indexing and return statistics

    This consistency enables easy database switching without code changes.

Supported Databases:
    - ChromaMultiTenancyIndexingPipeline: Local embedded with collection isolation
    - PineconeMultiTenancyIndexingPipeline: Managed cloud with namespace isolation
    - MilvusMultiTenancyIndexingPipeline: Distributed with partition isolation
    - QdrantMultiTenancyIndexingPipeline: High-performance with collection isolation
    - WeaviateMultiTenancyIndexingPipeline: Schema-based with collection isolation

Configuration Schema:
    Each pipeline requires a YAML configuration with:
        - Database connection parameters (API keys, URLs, paths)
        - Embedding model configuration (provider, model name, dimensions)
        - Dataloader settings (dataset, limit, preprocessing)
        - Tenant identification strategy (metadata field, naming convention)

Security Considerations:
    - Tenant IDs should be validated before indexing operations
    - Consider encrypting tenant data at rest where supported
    - Implement access controls to prevent cross-tenant operations
    - Audit logging for tenant data access and modifications

Example:
    >>> from vectordb.langchain.multi_tenancy.indexing import (
    ...     PineconeMultiTenancyIndexingPipeline,
    ... )
    >>> pipeline = PineconeMultiTenancyIndexingPipeline("configs/pinecone_mt.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents across tenants")

Note:
    Import specific pipelines directly from their respective modules rather
    than from this package-level __init__.py.
"""

from vectordb.langchain.multi_tenancy.indexing.chroma import (
    ChromaMultiTenancyIndexingPipeline,
)
from vectordb.langchain.multi_tenancy.indexing.milvus import (
    MilvusMultiTenancyIndexingPipeline,
)
from vectordb.langchain.multi_tenancy.indexing.pinecone import (
    PineconeMultiTenancyIndexingPipeline,
)
from vectordb.langchain.multi_tenancy.indexing.qdrant import (
    QdrantMultiTenancyIndexingPipeline,
)
from vectordb.langchain.multi_tenancy.indexing.weaviate import (
    WeaviateMultiTenancyIndexingPipeline,
)


__all__ = [
    "ChromaMultiTenancyIndexingPipeline",
    "MilvusMultiTenancyIndexingPipeline",
    "PineconeMultiTenancyIndexingPipeline",
    "QdrantMultiTenancyIndexingPipeline",
    "WeaviateMultiTenancyIndexingPipeline",
]
