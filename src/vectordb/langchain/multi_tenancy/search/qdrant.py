"""Qdrant multi-tenancy search pipeline (LangChain).

This module provides a search pipeline for Qdrant vector database with
multi-tenancy support. Searches are strictly scoped to a single tenant's
collection, ensuring complete data isolation between tenants.

Tenant Isolation Model:
    Each tenant's data is stored in a Qdrant collection named by tenant_id.
    This search pipeline:
    - Validates tenant_id at initialization
    - Embeds the query using the configured embedder
    - Queries ONLY within the tenant's collection
    - Returns results exclusively from that tenant's data

    This guarantees tenants cannot access each other's documents, even
    accidentally through query parameters or filters.

Search Flow:
    1. Validate tenant_id (cannot be empty)
    2. Load configuration and validate Qdrant settings
    3. Initialize embedder for query vectorization
    4. Initialize QdrantMultiTenancyPipeline for tenant operations
    5. Embed the search query
    6. Query Qdrant within tenant's collection only
    7. Optionally generate RAG answer using retrieved documents

Configuration Schema:
    Required:
        qdrant.url: Qdrant server URL
        tenant_id: Unique identifier passed to constructor
    Optional:
        qdrant.api_key: Qdrant API authentication (for cloud deployments)
        qdrant.collection_prefix: Prefix for tenant collections (default: "tenant_")
        embedder: Embedding model configuration
        rag: Optional LLM configuration for answer generation

Security Guarantees:
    - Search is scoped to collection=tenant_id
    - No cross-tenant data leakage possible
    - Metadata filters apply within tenant scope only
    - Results contain only tenant's own documents

Example:
    >>> from vectordb.langchain.multi_tenancy.search import (
    ...     QdrantMultiTenancySearchPipeline,
    ... )
    >>> # Search only within tenant "acme_corp" data
    >>> pipeline = QdrantMultiTenancySearchPipeline(
    ...     "config.yaml",
    ...     tenant_id="acme_corp",
    ... )
    >>> results = pipeline.search("quarterly revenue", top_k=5)
    >>> print(f"Found {len(results['documents'])} documents for {results['tenant_id']}")

See Also:
    - vectordb.langchain.multi_tenancy.indexing.qdrant: Tenant-scoped indexing
    - vectordb.langchain.multi_tenancy.qdrant: Core multi-tenancy implementation
"""

import logging
from typing import Any

from vectordb.langchain.multi_tenancy.qdrant import QdrantMultiTenancyPipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class QdrantMultiTenancySearchPipeline:
    """Qdrant search pipeline for multi-tenant scenarios (LangChain).

    Performs semantic search within a single tenant's isolated collection,
    ensuring complete data segregation. Returns documents only from the
    specified tenant's data.

    Attributes:
        config: Validated configuration dictionary containing Qdrant settings,
            embedder configuration, and optional LLM configuration.
        tenant_id: Unique identifier for the tenant. Used as collection name.
        embedder: LangChain embedder instance for query vectorization.
        pipeline: QdrantMultiTenancyPipeline for tenant-specific operations.
        llm: Optional LangChain LLM for RAG answer generation.

    Design Decisions:
        - Collection isolation: Uses Qdrant collections for tenant isolation,
          which is efficient and secure.
        - Fail-fast validation: tenant_id is validated at initialization to
          catch errors early.
        - Tenant-scoped results: All returned documents belong to the tenant.

    Example:
        >>> config = {
        ...     "qdrant": {
        ...         "url": "http://localhost:6333",
        ...         "collection_prefix": "tenant_",
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = QdrantMultiTenancySearchPipeline(config, tenant_id="tenant_1")
        >>> results = pipeline.search("machine learning", top_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str, tenant_id: str) -> None:
        """Initialize the multi-tenancy search pipeline.

        Validates tenant_id, loads configuration, initializes the embedder,
        and sets up the Qdrant multi-tenancy pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'qdrant' section with URL details.
            tenant_id: Unique identifier for the tenant. Search will be scoped
                to this tenant's collection. Cannot be empty.

        Raises:
            ValueError: If tenant_id is empty or None, or if required Qdrant
                configuration (url) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            The tenant_id must match the collection used during indexing for
            this tenant. Mismatched tenant_ids will return empty results.
        """
        # Validate tenant_id early to fail fast on invalid input.
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")
        self.tenant_id = tenant_id

        # Initialize embedder for query vectorization.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize the core multi-tenancy pipeline.
        qdrant_config = self.config["qdrant"]
        self.pipeline = QdrantMultiTenancyPipeline(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
            collection_prefix=qdrant_config.get("collection_prefix", "tenant_"),
        )

        # Initialize optional LLM for RAG.
        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Qdrant multi-tenancy search pipeline for tenant: %s",
            tenant_id,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search within the tenant's isolated collection.

        Embeds the query and retrieves documents exclusively from the tenant's
        collection. This ensures complete data isolation - results contain only
        documents belonging to this tenant.

        The search process:
            1. Embed query text using configured embedder
            2. Query Qdrant within tenant's collection only
            3. Apply optional metadata filters (within tenant scope)
            4. Optionally generate RAG answer using retrieved documents

        Args:
            query: Search query text to embed and match against documents.
            top_k: Number of documents to return. Default is 10.
            filters: Optional metadata filters to apply during retrieval.
                Filters are scoped to the tenant's collection only.
                Example: {"category": "technology", "status": "active"}

        Returns:
            Dictionary containing:
                - documents: List of Document objects from tenant's data
                - query: Original query string
                - tenant_id: The tenant ID that was searched
                - answer: Generated RAG answer if LLM configured (optional)

        Raises:
            ValueError: If query is empty or invalid.
            RuntimeError: If embedding generation or Qdrant query fails.

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

        # Get the tenant-specific collection name.
        collection_name = self.pipeline._get_tenant_collection_name(self.tenant_id)

        # Query Qdrant within tenant-specific collection only.
        # This is the key isolation mechanism - collection_name=tenant_id
        documents = self.pipeline.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=collection_name,
        )
        logger.info(
            "Retrieved %d documents for tenant %s from Qdrant",
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
