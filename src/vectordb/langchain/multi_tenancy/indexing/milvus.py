"""Milvus multi-tenancy indexing pipeline (LangChain).

This module provides an indexing pipeline for Milvus vector database with
multi-tenancy support. Multi-tenancy ensures complete data isolation between
different tenants (customers, organizations, users) using Milvus's partition
mechanism.

Multi-Tenancy Strategy:
    Milvus partitions provide perfect tenant isolation:
    - Each tenant's data is stored in a separate partition named by tenant_id
    - Queries are scoped to a single partition, preventing cross-tenant access
    - Metadata filters can further restrict within a tenant's data
    - Tenant deletion is implemented as partition deletion

    This approach ensures:
    - Data isolation: Tenants cannot access each other's documents
    - Query performance: Partition filtering is efficient in Milvus
    - Scalability: Each partition scales independently
    - Compliance: Meets strict data segregation requirements

Pipeline Flow:
    1. Validate tenant_id (cannot be empty)
    2. Load configuration and validate Milvus settings
    3. Initialize embedder for dense vector generation
    4. Initialize MilvusMultiTenancyPipeline for tenant operations
    5. Load documents from configured data source
    6. Generate embeddings for all documents
    7. Index documents to tenant-specific partition

Configuration Schema:
    Required:
        milvus.host: Milvus server host
        milvus.port: Milvus server port
        tenant_id: Unique identifier passed to constructor
    Optional:
        milvus.collection_name: Collection name (default: "multi_tenancy")
        milvus.dimension: Vector dimension (default: 384)
        embedder: Embedding model configuration
        dataloader: Data source configuration

Tenant Isolation Guarantees:
    - Documents are indexed to partition=tenant_id
    - Searches only query within the tenant's partition
    - Delete tenant removes entire partition
    - No cross-tenant data leakage possible

Example:
    >>> from vectordb.langchain.multi_tenancy.indexing import (
    ...     MilvusMultiTenancyIndexingPipeline,
    ... )
    >>> # Index documents for tenant "acme_corp"
    >>> pipeline = MilvusMultiTenancyIndexingPipeline(
    ...     "config.yaml",
    ...     tenant_id="acme_corp",
    ... )
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} docs for {result['tenant_id']}")

See Also:
    - vectordb.langchain.multi_tenancy.search.milvus: Tenant-scoped search
    - vectordb.langchain.multi_tenancy.milvus: Core multi-tenancy implementation
"""

import logging
from typing import Any

from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.multi_tenancy.milvus import MilvusMultiTenancyPipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class MilvusMultiTenancyIndexingPipeline:
    """Milvus indexing pipeline for multi-tenant scenarios (LangChain).

    Indexes documents for a specific tenant in an isolated partition,
    ensuring complete data segregation between tenants. Uses Milvus's
    partition mechanism for efficient tenant isolation.

    Attributes:
        config: Validated configuration dictionary containing Milvus settings,
            embedder configuration, and data source details.
        tenant_id: Unique identifier for the tenant. Used as partition name.
        embedder: LangChain embedder instance for generating document vectors.
        pipeline: MilvusMultiTenancyPipeline for tenant-specific operations.

    Design Decisions:
        - Partition-per-tenant: Each tenant gets a dedicated partition named
          by their tenant_id. This is the most secure isolation model.
        - Shared collection: All tenants share one Milvus collection, with
          partitions providing isolation. This is cost-effective and easier
          to manage.
        - Tenant validation: tenant_id is validated at initialization to fail
          fast on invalid input.

    Example:
        >>> config = {
        ...     "milvus": {
        ...         "host": "localhost",
        ...         "port": 19530,
        ...         "collection_name": "multi-tenant-docs",
        ...         "dimension": 384,
        ...     },
        ...     "embedder": {"model_name": "all-MiniLM-L6-v2"},
        ... }
        >>> pipeline = MilvusMultiTenancyIndexingPipeline(config, tenant_id="tenant_1")
        >>> result = pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str, tenant_id: str) -> None:
        """Initialize the multi-tenancy indexing pipeline.

        Validates tenant_id, loads configuration, initializes the embedder,
        and sets up the Milvus multi-tenancy pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain 'milvus' section with host and port details.
            tenant_id: Unique identifier for the tenant. This will be used as
                the partition name in Milvus. Cannot be empty.

        Raises:
            ValueError: If tenant_id is empty or None, or if required Milvus
                configuration (host, port) is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            The tenant_id becomes the partition name in Milvus. Choose
            tenant_ids that are valid partition names (no special characters
            that Milvus doesn't support in partition names).
        """
        # Validate tenant_id early to fail fast on invalid input.
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")
        self.tenant_id = tenant_id

        # Initialize embedder for dense vector generation.
        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize the core multi-tenancy pipeline.
        milvus_config = self.config["milvus"]
        self.pipeline = MilvusMultiTenancyPipeline(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
            collection_name=milvus_config.get("collection_name", "multi_tenancy"),
            dimension=milvus_config.get("dimension", 384),
        )

        logger.info(
            "Initialized Milvus multi-tenancy indexing pipeline for tenant: %s",
            tenant_id,
        )

    def run(self) -> dict[str, Any]:
        """Execute the multi-tenancy indexing pipeline for the tenant.

        Loads documents from the configured data source, generates embeddings,
        and indexes documents to the tenant's isolated partition.

        The pipeline follows this sequence:
            1. Load documents from configured dataloader
            2. Generate embeddings for each document
            3. Index documents to tenant-specific partition

        Returns:
            Dictionary with operation statistics:
                - documents_indexed: Number of documents successfully indexed (int)
                - tenant_id: The tenant ID that was indexed (str)

        Raises:
            RuntimeError: If embedding generation or Milvus upsert fails.
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
            "Indexed %d documents for tenant %s to Milvus",
            num_indexed,
            self.tenant_id,
        )

        return {"documents_indexed": num_indexed, "tenant_id": self.tenant_id}
