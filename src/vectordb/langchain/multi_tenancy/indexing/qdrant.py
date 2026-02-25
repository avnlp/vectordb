"""Qdrant multi-tenancy indexing pipeline (LangChain).

This module provides an indexing pipeline for Qdrant vector database with
multi-tenancy support. Multi-tenancy ensures complete data isolation between
different tenants (customers, organizations, users) using Qdrant's collection
mechanism.

Multi-Tenancy Strategy:
    Qdrant collections provide perfect tenant isolation:
    - Each tenant's data is stored in a separate collection named by tenant_id
    - Queries are scoped to a single collection, preventing cross-tenant access
    - Metadata filters can further restrict within a tenant's data
    - Tenant deletion is implemented as collection deletion

    This approach ensures:
    - Data isolation: Tenants cannot access each other's documents
    - Query performance: Collection filtering is efficient in Qdrant
    - Scalability: Each collection scales independently
    - Compliance: Meets strict data segregation requirements

Pipeline Flow:
    1. Validate tenant_id (cannot be empty)
    2. Load configuration and validate Qdrant settings
    3. Initialize embedder for dense vector generation
    4. Initialize QdrantMultiTenancyPipeline for tenant operations
    5. Load documents from configured data source
    6. Generate embeddings for all documents
    7. Index documents to tenant-specific collection

Configuration Schema:
    Required:
        qdrant.url: Qdrant server URL
        tenant_id: Unique identifier passed to constructor
    Optional:
        qdrant.api_key: Qdrant API authentication (for cloud deployments)
        qdrant.collection_prefix: Prefix for tenant collections (default: "tenant_")
        embedder: Embedding model configuration
        dataloader: Data source configuration

Tenant Isolation Guarantees:
    - Documents are indexed to collection=tenant_id
    - Searches only query within the tenant's collection
    - Delete tenant removes entire collection
    - No cross-tenant data leakage possible

Example:
    >>> from vectordb.langchain.multi_tenancy.indexing import (
    ...     QdrantMultiTenancyIndexingPipeline,
    ... )
    >>> # Index documents for tenant "acme_corp"
    >>> pipeline = QdrantMultiTenancyIndexingPipeline(
    ...     "config.yaml",
    ...     tenant_id="acme_corp",
    ... )
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} docs for {result['tenant_id']}")

See Also:
    - vectordb.langchain.multi_tenancy.search.qdrant: Tenant-scoped search
    - vectordb.langchain.multi_tenancy.qdrant: Core multi-tenancy implementation
"""

import logging
from typing import Any

from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.multi_tenancy.qdrant import QdrantMultiTenancyPipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class QdrantMultiTenancyIndexingPipeline:
    """Qdrant indexing pipeline for multi-tenant scenarios (LangChain).

        Indexes documents for a specific tenant in an isolated collection,
        ensuring complete data segregation between tenants. Uses Qdrant's
    collection mechanism for efficient tenant isolation.

    Attributes:
            config: Validated configuration dictionary containing Qdrant settings,
                embedder configuration, and data source details.
            tenant_id: Unique identifier for the tenant. Used as collection name.
            embedder: LangChain embedder instance for generating document vectors.
            pipeline: QdrantMultiTenancyPipeline for tenant-specific operations.

        Design Decisions:
            - Collection-per-tenant: Each tenant gets a dedicated collection named
              by their tenant_id. This is the most secure isolation model.
            - Collection prefix: Optional prefix can be added to all tenant
              collections for easier management and identification.
            - Tenant validation: tenant_id is validated at initialization to fail
              fast on invalid input.

    Example:
            >>> config = {
            ...     "qdrant": {
            ...         "url": "http://localhost:6333",
            ...         "collection_prefix": "tenant_",
            ...     },
            ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
            ... }
            >>> pipeline = QdrantMultiTenancyIndexingPipeline(
            ...     config, tenant_id="tenant_1"
            ... )
            >>> result = pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str, tenant_id: str) -> None:
        """Initialize the multi-tenancy indexing pipeline.

        Validates tenant_id, loads configuration, initializes the embedder,
        and sets up the Qdrant multi-tenancy pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'qdrant' section with URL details.
            tenant_id: Unique identifier for the tenant. This will be used as
                the collection name in Qdrant. Cannot be empty.

        Raises:
            ValueError: If tenant_id is empty or None, or if required Qdrant
                configuration (url) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            The tenant_id becomes part of the collection name in Qdrant.
            If collection_prefix is set, the full collection name will be
            "{prefix}{tenant_id}". Choose tenant_ids that are valid collection
            names (no special characters that Qdrant doesn't support).
        """
        # Validate tenant_id early to fail fast on invalid input.
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")
        self.tenant_id = tenant_id

        # Initialize embedder for dense vector generation.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize the core multi-tenancy pipeline.
        qdrant_config = self.config["qdrant"]
        self.pipeline = QdrantMultiTenancyPipeline(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
            collection_prefix=qdrant_config.get("collection_prefix", "tenant_"),
        )

        logger.info(
            "Initialized Qdrant multi-tenancy indexing pipeline for tenant: %s",
            tenant_id,
        )

    def run(self) -> dict[str, Any]:
        """Execute the multi-tenancy indexing pipeline for the tenant.

        Loads documents from the configured data source, generates embeddings,
        and indexes documents to the tenant's isolated collection.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Index documents to tenant-specific collection

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)
                - tenant_id: The tenant ID that was indexed (str)

        Raises:
            RuntimeError: If embedding generation or Qdrant upsert fails.
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

        # Index documents for tenant
        num_indexed = self.pipeline.index_for_tenant(
            tenant_id=self.tenant_id,
            documents=docs,
            embeddings=embeddings,
        )

        logger.info(
            "Indexed %d documents for tenant %s to Qdrant",
            num_indexed,
            self.tenant_id,
        )

        return {"documents_indexed": num_indexed, "tenant_id": self.tenant_id}
