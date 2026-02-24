"""Weaviate multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any

from haystack import Document

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.common.embeddings import (
    create_document_embedder,
    truncate_embeddings,
)
from vectordb.haystack.multi_tenancy.common.types import (
    MultitenancyTimingMetrics,
    TenantIndexResult,
    TenantIsolationStrategy,
    TenantStats,
    TenantStatus,
)


logger = logging.getLogger(__name__)


class WeaviateMultitenancyIndexingPipeline(BaseMultitenancyPipeline):
    """Weaviate indexing pipeline with native multi-tenancy.

    Weaviate provides native multi-tenancy through:
    - Per-tenant shards/collections
    - Tenant-specific operations
    - Built-in tenant isolation
    """

    def __init__(
        self,
        config_path: str,
        tenant_context: Any = None,
    ) -> None:
        """Initialize Weaviate indexing pipeline.

        Args:
            config_path: Path to YAML configuration file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self._client: Any = None
        self._embedder: Any = None
        super().__init__(config_path=config_path, tenant_context=tenant_context)

    def _connect(self) -> None:
        """Connect to Weaviate and initialize embedder."""
        import weaviate

        weaviate_config = self.config.get("weaviate", self.config.get("database", {}))

        # Connect to Weaviate client
        auth_config = None
        if "api_key" in weaviate_config:
            from weaviate.auth import AuthApiKey

            auth_config = AuthApiKey(api_key=weaviate_config["api_key"])

        additional_headers = {}
        if "headers" in weaviate_config:
            additional_headers.update(weaviate_config["headers"])

        self._client = weaviate.Client(
            url=weaviate_config.get(
                "url", os.environ.get("WEAVIATE_URL", "http://localhost:8080")
            ),
            auth_client_secret=auth_config,
            additional_headers=additional_headers,
        )

        self._embedder = create_document_embedder(self.config)
        self._embedder.warm_up()

    def _get_class_name(self) -> str:
        """Get class name from config."""
        return self.config.get("collection", {}).get("name", "MultiTenancy")

    def close(self) -> None:
        """Close Weaviate connection."""
        if self._client:
            self._client.close()

    def _get_document_store(self) -> Any:
        """Return configured Weaviate client.

        Returns:
            Weaviate Client instance or None if not initialized.
        """
        return self._client

    @property
    def isolation_strategy(self) -> TenantIsolationStrategy:
        """Return the isolation strategy for Weaviate.

        Returns:
            TenantIsolationStrategy.NATIVE_MULTITENANCY
        """
        return TenantIsolationStrategy.NATIVE_MULTITENANCY

    def create_tenant(self, tenant_id: str) -> bool:
        """Create a new tenant in Weaviate.

        Weaviate uses native multi-tenancy with per-tenant shards.

        Args:
            tenant_id: Tenant identifier to create.

        Returns:
            True if tenant was created, False if already exists.
        """
        class_name = self._get_class_name()

        try:
            tenants = self._client.schema.get_tenant(class_name)
            existing_tenant_names = [t["name"] for t in tenants]
            if tenant_id in existing_tenant_names:
                return False

            self._client.schema.add_tenant(class_name, tenant_id)
            logger.info(f"Created tenant {tenant_id} in class {class_name}")
            return True
        except Exception:
            # If getting tenants fails, try to create the class with multi-tenancy
            try:
                self._ensure_class_exists(class_name)
                self._client.schema.add_tenant(class_name, tenant_id)
                logger.info(f"Created tenant {tenant_id} in class {class_name}")
                return True
            except Exception as e:
                logger.error(f"Error creating tenant: {e}")
                return False

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists in Weaviate.

        Args:
            tenant_id: Tenant identifier to check.

        Returns:
            True if tenant exists, False otherwise.
        """
        if self._client is None:
            return False

        class_name = self._get_class_name()

        try:
            tenants = self._client.schema.get_tenant(class_name)
            existing_tenant_names = [t["name"] for t in tenants]
            return tenant_id in existing_tenant_names
        except Exception:
            return False

    def get_tenant_stats(self, tenant_id: str) -> TenantStats:
        """Get statistics for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            TenantStats with tenant statistics.
        """
        if self._client is None:
            return TenantStats(
                tenant_id=tenant_id,
                document_count=0,
                vector_count=0,
                status=TenantStatus.UNKNOWN,
            )

        try:
            # Query for count of objects in this tenant
            class_name = self._get_class_name()
            result = (
                self._client.query.aggregate(class_name)
                .with_meta_count()
                .with_tenant(tenant_id)
                .do()
            )

            count = 0
            if "data" in result and "Aggregate" in result["data"]:
                agg_result = result["data"]["Aggregate"].get(class_name, [])
                if agg_result:
                    count = agg_result[0].get("meta", {}).get("count", 0)

            return TenantStats(
                tenant_id=tenant_id,
                document_count=count,
                vector_count=count,
                status=TenantStatus.ACTIVE,
            )
        except Exception:
            return TenantStats(
                tenant_id=tenant_id,
                document_count=0,
                vector_count=0,
                status=TenantStatus.UNKNOWN,
            )

    def list_tenants(self) -> list[str]:
        """List all tenants in Weaviate.

        Returns:
            List of tenant identifiers.
        """
        if self._client is None:
            return []

        class_name = self._get_class_name()

        try:
            tenants = self._client.schema.get_tenant(class_name)
            return [t["name"] for t in tenants]
        except Exception:
            return []

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all its data.

        Args:
            tenant_id: Tenant identifier to delete.

        Returns:
            True if tenant was deleted, False if not found.
        """
        if self._client is None:
            return False

        class_name = self._get_class_name()

        try:
            # Delete all objects for this tenant first
            self._client.batch.delete_objects(
                class_name=class_name,
                where={
                    "path": ["tenant_id"],
                    "operator": "Equal",
                    "valueString": tenant_id,
                },
                tenant=tenant_id,
            )

            # Then delete the tenant itself
            self._client.schema.delete_tenant(class_name, tenant_id)
            return True
        except Exception:
            return False

    def index_documents(
        self,
        documents: list[Document],
        tenant_id: str | None = None,
    ) -> int:
        """Index documents for a tenant.

        Args:
            documents: List of Haystack Documents to index.
            tenant_id: Target tenant. Uses current context if None.

        Returns:
            Number of documents indexed.
        """
        target_tenant = tenant_id or self.tenant_context.tenant_id

        if not documents:
            self.logger.warning("No documents to index")
            return 0

        # Embed documents
        result = self._embedder.run(documents=documents)
        embedded_docs = result["documents"]

        # Truncate if output_dimension specified
        output_dim = self.config.get("embedding", {}).get("output_dimension")
        embedded_docs = truncate_embeddings(embedded_docs, output_dim)

        class_name = self._get_class_name()
        self._ensure_tenant_exists(class_name, target_tenant)

        # Prepare batch for insertion
        self._client.batch.configure(batch_size=100)
        with self._client.batch as batch:
            for doc in embedded_docs:
                properties = {
                    "content": doc.content or "",
                    "tenant_id": target_tenant,
                }
                for key, value in doc.meta.items():
                    if key != "tenant_id":  # Skip tenant_id to avoid duplication
                        properties[key] = (
                            str(value)
                            if not isinstance(value, (str, int, float, bool))
                            else value
                        )

                batch.add_data_object(
                    data_object=properties,
                    class_name=class_name,
                    vector=doc.embedding,
                    tenant=target_tenant,
                )

        self.logger.info(
            "Indexed %d documents for tenant %s",
            len(embedded_docs),
            target_tenant,
        )

        return len(embedded_docs)

    def query(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
    ) -> list[Document]:
        """Query documents within tenant scope.

        Args:
            query: Query string.
            top_k: Number of results to return.
            tenant_id: Target tenant. Uses current context if None.

        Returns:
            List of retrieved Documents.
        """
        target_tenant = tenant_id or self.tenant_context.tenant_id

        if self._client is None:
            return []

        # Embed query
        from vectordb.haystack.multi_tenancy.common.embeddings import (
            create_query_embedder,
        )

        query_embedder = create_query_embedder(self.config)
        query_embedder.warm_up()
        query_result = query_embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Query Weaviate with tenant
        class_name = self._get_class_name()
        results = (
            self._client.query.get(
                class_name,
                ["content", "tenant_id"],
            )
            .with_near_vector({"vector": query_embedding})
            .with_limit(top_k)
            .with_tenant(target_tenant)
            .do()
        )

        # Convert to Haystack Documents
        haystack_docs = []
        if "data" in results and "Get" in results["data"]:
            for item in results["data"]["Get"].get(class_name, []):
                doc = Document(
                    content=item.get("content", ""),
                    meta={"tenant_id": item.get("tenant_id", target_tenant)},
                )
                haystack_docs.append(doc)

        return haystack_docs

    def run(
        self,
        documents: list[Document] | None = None,
        tenant_id: str | None = None,
    ) -> TenantIndexResult:
        """Index documents for a tenant (convenience method).

        This method provides a convenient way to index documents while
        capturing timing metrics. For simple indexing without metrics,
        use index_documents() directly.

        Args:
            documents: Documents to index. If None, loads from dataloader.
            tenant_id: Target tenant. Uses context if None.

        Returns:
            TenantIndexResult with indexing metrics.
        """
        from vectordb.haystack.multi_tenancy.base import Timer

        target_tenant = tenant_id or self.tenant_context.tenant_id

        with Timer() as timer:
            if documents is None:
                documents = self._load_documents_from_dataloader()

            if not documents:
                self.logger.warning("No documents to index")
                return TenantIndexResult(
                    tenant_id=target_tenant,
                    documents_indexed=0,
                    collection_name=self._get_class_name(),
                    timing=MultitenancyTimingMetrics(
                        tenant_resolution_ms=0.0,
                        index_operation_ms=timer.elapsed_ms,
                        retrieval_ms=0.0,
                        total_ms=timer.elapsed_ms,
                        tenant_id=target_tenant,
                        num_documents=0,
                    ),
                )

            num_indexed = self.index_documents(documents, tenant_id)

        return TenantIndexResult(
            tenant_id=target_tenant,
            documents_indexed=num_indexed,
            collection_name=self._get_class_name(),
            timing=MultitenancyTimingMetrics(
                tenant_resolution_ms=0.0,
                index_operation_ms=timer.elapsed_ms,
                retrieval_ms=0.0,
                total_ms=timer.elapsed_ms,
                tenant_id=target_tenant,
                num_documents=num_indexed,
            ),
        )

    def _ensure_class_exists(self, class_name: str) -> None:
        """Ensure the Weaviate class exists with multi-tenancy enabled.

        Args:
            class_name: Name of the class to create.
        """
        try:
            # Check if class exists
            self._client.schema.get(class_name)
        except Exception:
            # Create the class with multi-tenancy
            class_obj = {
                "class": class_name,
                "vectorizer": "none",  # We provide vectors manually
                "multiTenancyConfig": {"enabled": True},
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "tenant_id", "dataType": ["string"]},
                ],
            }
            self._client.schema.create_class(class_obj)
            logger.info(f"Created class {class_name} with multi-tenancy enabled")

    def _ensure_tenant_exists(self, class_name: str, tenant_name: str) -> None:
        """Ensure the tenant exists in Weaviate.

        Args:
            class_name: Name of the class to add tenant to.
            tenant_name: Name of the tenant to create.
        """
        try:
            tenants = self._client.schema.get_tenant(class_name)
            existing_tenant_names = [t["name"] for t in tenants]
            if tenant_name not in existing_tenant_names:
                self._client.schema.add_tenant(class_name, tenant_name)
                logger.info(f"Created tenant {tenant_name} in class {class_name}")
        except Exception:
            # If getting tenants fails, create the class first
            self._ensure_class_exists(class_name)
            self._client.schema.add_tenant(class_name, tenant_name)
            logger.info(f"Created tenant {tenant_name} in class {class_name}")

    def _load_documents_from_dataloader(self) -> list[Document]:
        """Load documents from configured dataloader.

        Returns:
            List of Haystack Documents loaded from dataloader.
        """
        dataloader_config = self.config.get("dataloader", {})
        dataset_type = dataloader_config.get("dataset", "triviaqa")
        params = dataloader_config.get("params", {})

        dataset_id = params.get("dataset_name")
        split = params.get("split", "test")
        limit = params.get("limit")

        loader = DataloaderCatalog.create(
            dataset_type,
            split=split,
            limit=limit,
            dataset_id=dataset_id,
        )

        return loader.load().to_haystack()
