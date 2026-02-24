"""Weaviate multi-tenancy indexing pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from haystack import Document

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.multi_tenancy.common.config import load_config
from vectordb.haystack.multi_tenancy.common.embeddings import (
    create_document_embedder,
    truncate_embeddings,
)
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.timing import Timer
from vectordb.haystack.multi_tenancy.common.types import TenantIndexResult


logger = logging.getLogger(__name__)


class WeaviateMultitenancyIndexingPipeline:
    """Weaviate indexing pipeline with native multi-tenancy.

    Weaviate provides native multi-tenancy through:
    - Per-tenant shards/collections
    - Tenant-specific operations
    - Built-in tenant isolation
    """

    def __init__(
        self,
        config_or_path: dict[str, Any] | str | Path,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize Weaviate indexing pipeline.

        Args:
            config_or_path: Config dict or path to YAML file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self.config = load_config(config_or_path)
        self.tenant_context = TenantContext.resolve(tenant_context, self.config)
        self._client = None
        self._embedder = None
        self._connect()

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

    def run(
        self,
        documents: list[Document] | None = None,
        tenant_id: str | None = None,
    ) -> TenantIndexResult:
        """Index documents for a tenant.

        Args:
            documents: Documents to index. If None, loads from dataloader.
            tenant_id: Target tenant. Uses context if None.

        Returns:
            TenantIndexResult with indexing metrics.
        """
        target_tenant = tenant_id or self.tenant_context.tenant_id

        with Timer() as timer:
            if documents is None:
                documents = self._load_documents_from_dataloader()

            if not documents:
                logger.warning("No documents to index")
                return TenantIndexResult(
                    tenant_id=target_tenant,
                    documents_indexed=0,
                    collection_name=self._get_class_name(),
                    timing=self._create_timing_metrics(0, timer.elapsed_ms),
                )

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
                        tenant=target_tenant,  # This is the key for Weaviate multi-tenancy
                    )

        metrics = self._create_timing_metrics(len(embedded_docs), timer.elapsed_ms)

        result = TenantIndexResult(
            tenant_id=target_tenant,
            documents_indexed=len(embedded_docs),
            collection_name=class_name,
            timing=metrics,
        )

        logger.info(
            "Indexed %d documents for tenant %s in %.1fms",
            len(embedded_docs),
            target_tenant,
            timer.elapsed_ms,
        )

        return result

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
            # If getting tenants fails, try to create the class with multi-tenancy
            try:
                # Define the class schema with multi-tenancy enabled
                class_obj = {
                    "class": class_name,
                    "vectorizer": "none",  # We'll provide vectors manually
                    "multiTenancyConfig": {"enabled": True},
                    "properties": [
                        {"name": "content", "dataType": ["text"]},
                        {"name": "tenant_id", "dataType": ["string"]},
                    ],
                }

                self._client.schema.create_class(class_obj)
                # Now add the tenant
                self._client.schema.add_tenant(class_name, tenant_name)
                logger.info(
                    f"Created class {class_name} with multi-tenancy and tenant {tenant_name}"
                )
            except Exception as e:
                logger.error(f"Error creating class or tenant: {e}")

    def _create_timing_metrics(
        self,
        num_documents: int,
        total_ms: float,
    ) -> Any:
        """Create timing metrics for indexing operation.

        Args:
            num_documents: Number of documents processed.
            total_ms: Total operation time in milliseconds.

        Returns:
            Timing metrics object.
        """
        return type(
            "TimingMetrics",
            (),
            {
                "tenant_resolution_ms": 0.0,
                "index_operation_ms": total_ms,
                "retrieval_ms": 0.0,
                "total_ms": total_ms,
                "tenant_id": self.tenant_context.tenant_id,
                "num_documents": num_documents,
            },
        )()
