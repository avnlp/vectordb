"""Pinecone multi-tenancy indexing pipeline (LangChain).

This module provides an indexing pipeline for Pinecone vector database with
multi-tenancy support. Multi-tenancy ensures complete data isolation between
different tenants (customers, organizations, users) using Pinecone's namespace
mechanism.

Multi-Tenancy Strategy:
    Pinecone namespaces provide perfect tenant isolation:
    - Each tenant's data is stored in a separate namespace named by tenant_id
    - Queries are scoped to a single namespace, preventing cross-tenant access
    - Metadata filters can further restrict within a tenant's data
    - Tenant deletion is implemented as namespace deletion

    This approach ensures:
    - Data isolation: Tenants cannot access each other's documents
    - Query performance: Namespace filtering is efficient in Pinecone
    - Scalability: Each namespace scales independently
    - Compliance: Meets strict data segregation requirements

Pipeline Flow:
    1. Validate tenant_id (cannot be empty)
    2. Load configuration and validate Pinecone settings
    3. Initialize embedder for dense vector generation
    4. Initialize PineconeMultiTenancyPipeline for tenant operations
    5. Load documents from configured data source
    6. Generate embeddings for all documents
    7. Create Pinecone index (shared across all tenants)
    8. Index documents to tenant-specific namespace

Configuration Schema:
    Required:
        pinecone.api_key: Pinecone API authentication
        pinecone.index_name: Shared index name for all tenants
        tenant_id: Unique identifier passed to constructor
    Optional:
        pinecone.dimension: Vector dimension (default: 384)
        pinecone.metric: Distance metric (default: "cosine")
        pinecone.recreate: Whether to recreate index (default: False)
        embedder: Embedding model configuration
        dataloader: Data source configuration

Tenant Isolation Guarantees:
    - Documents are indexed to namespace=tenant_id
    - Searches only query within the tenant's namespace
    - Delete tenant removes entire namespace
    - No cross-tenant data leakage possible

Example:
    >>> from vectordb.langchain.multi_tenancy.indexing import (
    ...     PineconeMultiTenancyIndexingPipeline,
    ... )
    >>> # Index documents for tenant "acme_corp"
    >>> pipeline = PineconeMultiTenancyIndexingPipeline(
    ...     "config.yaml",
    ...     tenant_id="acme_corp",
    ... )
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} docs for {result['tenant_id']}")

See Also:
    - vectordb.langchain.multi_tenancy.search.pinecone: Tenant-scoped search
    - vectordb.langchain.multi_tenancy.pinecone: Core multi-tenancy implementation
    - vectordb.langchain.multi_tenancy.base: Abstract base class interface
"""

import logging
from typing import Any

from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.multi_tenancy.pinecone import PineconeMultiTenancyPipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class PineconeMultiTenancyIndexingPipeline:
    """Pinecone indexing pipeline for multi-tenant scenarios (LangChain).

    Indexes documents for a specific tenant in an isolated namespace,
    ensuring complete data segregation between tenants. Uses Pinecone's
    namespace mechanism for efficient tenant isolation.

    Attributes:
        config: Validated configuration dictionary containing Pinecone settings,
            embedder configuration, and data source details.
        tenant_id: Unique identifier for the tenant. Used as namespace name.
        embedder: LangChain embedder instance for generating document vectors.
        pipeline: PineconeMultiTenancyPipeline for tenant-specific operations.
        index_name: Name of the shared Pinecone index.
        dimension: Vector dimension matching the embedder output.

    Design Decisions:
        - Namespace-per-tenant: Each tenant gets a dedicated namespace named
          by their tenant_id. This is the most secure isolation model.
        - Shared index: All tenants share one Pinecone index, with namespaces
          providing isolation. This is cost-effective and easier to manage.
        - Tenant validation: tenant_id is validated at initialization to fail
          fast on invalid input.

    Example:
        >>> config = {
        ...     "pinecone": {
        ...         "api_key": "pc-api-...",
        ...         "index_name": "multi-tenant-docs",
        ...         "dimension": 384,
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = PineconeMultiTenancyIndexingPipeline(
        ...     config, tenant_id="tenant_1"
        ... )
        >>> result = pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str, tenant_id: str) -> None:
        """Initialize the multi-tenancy indexing pipeline.

        Validates tenant_id, loads configuration, initializes the embedder,
        and sets up the Pinecone multi-tenancy pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'pinecone' section with API key and index details.
            tenant_id: Unique identifier for the tenant. This will be used as
                the namespace name in Pinecone. Cannot be empty.

        Raises:
            ValueError: If tenant_id is empty or None, or if required Pinecone
                configuration (api_key, index_name) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            The tenant_id becomes the namespace name in Pinecone. Choose
            tenant_ids that are valid namespace names (no special characters
            that Pinecone doesn't support in namespace names).
        """
        # Validate tenant_id early to fail fast on invalid input.
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")
        self.tenant_id = tenant_id

        # Initialize embedder for dense vector generation.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize the core multi-tenancy pipeline.
        pinecone_config = self.config["pinecone"]
        self.pipeline = PineconeMultiTenancyPipeline(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
            dimension=pinecone_config.get("dimension", 384),
        )

        # Store settings for pipeline operations.
        self.index_name = pinecone_config.get("index_name")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info(
            "Initialized Pinecone multi-tenancy indexing pipeline for tenant: %s",
            tenant_id,
        )

    def run(self) -> dict[str, Any]:
        """Execute the multi-tenancy indexing pipeline for the tenant.

        Loads documents from the configured data source, generates embeddings,
        creates the Pinecone index if needed, and indexes documents to the
        tenant's isolated namespace.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Create Pinecone index (shared across tenants)
            4. Index documents to tenant-specific namespace

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)
                - tenant_id: The tenant ID that was indexed (str)

        Raises:
            RuntimeError: If embedding generation or Pinecone upsert fails.
            ValueError: If no documents are found in the data source.

        Example:
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents")
            >>> print(f"For tenant: {result['tenant_id']}")
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_langchain()
        logger.info("Loaded %d documents for tenant %s", len(documents), self.tenant_id)

        if not documents:
            logger.warning("No documents to index for tenant: %s", self.tenant_id)
            return {"documents_indexed": 0, "tenant_id": self.tenant_id}

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info(
            "Generated embeddings for %d documents for tenant %s",
            len(docs),
            self.tenant_id,
        )

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.pipeline.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        # Index documents for tenant
        num_indexed = self.pipeline.index_for_tenant(
            tenant_id=self.tenant_id,
            documents=docs,
            embeddings=embeddings,
        )

        logger.info(
            "Indexed %d documents for tenant %s to Pinecone",
            num_indexed,
            self.tenant_id,
        )

        return {"documents_indexed": num_indexed, "tenant_id": self.tenant_id}
