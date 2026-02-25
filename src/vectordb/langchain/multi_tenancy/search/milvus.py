"""Milvus multi-tenancy search pipeline (LangChain).

This module provides a search pipeline for Milvus vector database with
multi-tenancy support. Searches are strictly scoped to a single tenant's
partition, ensuring complete data isolation between tenants.

Tenant Isolation Model:
    Each tenant's data is stored in a Milvus partition named by tenant_id.
    This search pipeline:
    - Validates tenant_id at initialization
    - Embeds the query using the configured embedder
    - Queries ONLY within the tenant's partition
    - Returns results exclusively from that tenant's data

    This guarantees tenants cannot access each other's documents, even
    accidentally through query parameters or filters.

Search Flow:
    1. Validate tenant_id (cannot be empty)
    2. Load configuration and validate Milvus settings
    3. Initialize embedder for query vectorization
    4. Initialize MilvusMultiTenancyPipeline for tenant operations
    5. Embed the search query
    6. Query Milvus within tenant's partition only
    7. Optionally generate RAG answer using retrieved documents

Configuration Schema:
    Required:
        milvus.host: Milvus server host
        milvus.port: Milvus server port
        tenant_id: Unique identifier passed to constructor
    Optional:
        milvus.collection_name: Collection name (default: "multi_tenancy")
        milvus.dimension: Vector dimension (default: 384)
        embedder: Embedding model configuration
        rag: Optional LLM configuration for answer generation

Security Guarantees:
    - Search is scoped to partition=tenant_id
    - No cross-tenant data leakage possible
    - Metadata filters apply within tenant scope only
    - Results contain only tenant's own documents

Example:
    >>> from vectordb.langchain.multi_tenancy.search import (
    ...     MilvusMultiTenancySearchPipeline,
    ... )
    >>> # Search only within tenant "acme_corp" data
    >>> pipeline = MilvusMultiTenancySearchPipeline(
    ...     "config.yaml",
    ...     tenant_id="acme_corp",
    ... )
    >>> results = pipeline.search("quarterly revenue", top_k=5)
    >>> print(f"Found {len(results['documents'])} documents for {results['tenant_id']}")

See Also:
    - vectordb.langchain.multi_tenancy.indexing.milvus: Tenant-scoped indexing
    - vectordb.langchain.multi_tenancy.milvus: Core multi-tenancy implementation
"""

import logging
from typing import Any

from vectordb.langchain.multi_tenancy.milvus import MilvusMultiTenancyPipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class MilvusMultiTenancySearchPipeline:
    """Milvus search pipeline for multi-tenant scenarios (LangChain).

    Performs semantic search within a single tenant's isolated partition,
    ensuring complete data segregation. Returns documents only from the
    specified tenant's data.

    Attributes:
        config: Validated configuration dictionary containing Milvus settings,
            embedder configuration, and optional LLM configuration.
        tenant_id: Unique identifier for the tenant. Used as partition name.
        embedder: LangChain embedder instance for query vectorization.
        pipeline: MilvusMultiTenancyPipeline for tenant-specific operations.
        llm: Optional LangChain LLM for RAG answer generation.

    Design Decisions:
        - Partition isolation: Uses Milvus partitions for tenant isolation,
          which is efficient and secure.
        - Fail-fast validation: tenant_id is validated at initialization to
          catch errors early.
        - Tenant-scoped results: All returned documents belong to the tenant.

    Example:
        >>> config = {
        ...     "milvus": {
        ...         "host": "localhost",
        ...         "port": 19530,
        ...         "collection_name": "multi-tenant-docs",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = MilvusMultiTenancySearchPipeline(config, tenant_id="tenant_1")
        >>> results = pipeline.search("machine learning", top_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str, tenant_id: str) -> None:
        """Initialize the multi-tenancy search pipeline.

        Validates tenant_id, loads configuration, initializes the embedder,
        and sets up the Milvus multi-tenancy pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'milvus' section with host and port details.
            tenant_id: Unique identifier for the tenant. Search will be scoped
                to this tenant's partition. Cannot be empty.

        Raises:
            ValueError: If tenant_id is empty or None, or if required Milvus
                configuration (host, port) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            The tenant_id must match the partition used during indexing for
            this tenant. Mismatched tenant_ids will return empty results.
        """
        # Validate tenant_id early to fail fast on invalid input.
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")
        self.tenant_id = tenant_id

        # Initialize embedder for query vectorization.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize the core multi-tenancy pipeline.
        milvus_config = self.config["milvus"]
        self.pipeline = MilvusMultiTenancyPipeline(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
            collection_name=milvus_config.get("collection_name", "multi_tenancy"),
            dimension=milvus_config.get("dimension", 384),
        )

        # Initialize optional LLM for RAG.
        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Milvus multi-tenancy search pipeline for tenant: %s",
            tenant_id,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search within the tenant's isolated partition.

        Embeds the query and retrieves documents exclusively from the tenant's
        partition. This ensures complete data isolation - results contain only
        documents belonging to this tenant.

        The search process:
            1. Embed query text using configured embedder
            2. Query Milvus within tenant's partition only
            3. Apply optional metadata filters (within tenant scope)
            4. Optionally generate RAG answer using retrieved documents

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of documents to return. Default is 10.
            filters: Optional metadata filters to apply during retrieval.
                Filters are scoped to the tenant's partition only.
                Example: {"category": "technology", "status": "active"}

        Returns:
            Dictionary containing:
                - documents: List of Document objects from tenant's data
                - query: Original query string
                - tenant_id: The tenant ID that was searched
                - answer: Generated RAG answer if LLM configured (optional)

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation or Milvus query fails.

        Example:
            >>> results = pipeline.search(
            ...     query="quarterly report",
            ...     top_k=5,
            ...     filters={"year": 2024},
            ... )
            >>> print(f"Found {len(results['documents'])} documents")
            >>> print(f"For tenant: {results['tenant_id']}")
        """
        # Generate query embedding for semantic search.
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query for tenant %s: %s", self.tenant_id, query[:50])

        # Query Milvus within tenant-specific partition only.
        # This is the key isolation mechanism - partition_name=self.tenant_id
        documents = self.pipeline.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=self.pipeline.collection_name,
            partition_name=self.tenant_id,
        )
        logger.info(
            "Retrieved %d documents for tenant %s from Milvus",
            len(documents),
            self.tenant_id,
        )

        result = {
            "documents": documents,
            "query": query,
            "tenant_id": self.tenant_id,
        }

        # Generate RAG answer if LLM is configured.
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer for tenant %s", self.tenant_id)

        return result
