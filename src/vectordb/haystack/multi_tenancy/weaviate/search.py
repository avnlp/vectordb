"""Weaviate multi-tenancy search pipeline with RAG support."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from haystack import Document, Pipeline

from vectordb.haystack.multi_tenancy.common.config import load_config
from vectordb.haystack.multi_tenancy.common.embeddings import (
    create_text_embedder,
)
from vectordb.haystack.multi_tenancy.common.rag import create_rag_pipeline
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.timing import Timer
from vectordb.haystack.multi_tenancy.common.types import (
    MultitenancyTimingMetrics,
    TenantQueryResult,
    TenantRAGResult,
    TenantRetrievalResult,
)


logger = logging.getLogger(__name__)


class WeaviateMultitenancySearchPipeline:
    """Weaviate search pipeline with native multi-tenancy.

    Provides both retrieval-only and RAG capabilities with tenant isolation.
    """

    def __init__(
        self,
        config_or_path: dict[str, Any] | str | Path,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize Weaviate search pipeline.

        Args:
            config_or_path: Config dict or path to YAML file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self.config = load_config(config_or_path)
        self.tenant_context = TenantContext.resolve(tenant_context, self.config)
        self._client = None
        self._embedder = None
        self._rag_pipeline: Pipeline | None = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Weaviate and initialize components."""
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

        self._embedder = create_text_embedder(self.config)
        self._embedder.warm_up()

        if self.config.get("rag", {}).get("enabled", False):
            self._rag_pipeline = create_rag_pipeline(self.config)

    def _get_class_name(self) -> str:
        """Get class name from config."""
        return self.config.get("collection", {}).get("name", "MultiTenancy")

    def close(self) -> None:
        """Close Weaviate connection."""
        if self._client:
            self._client.close()

    def query(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
    ) -> TenantRetrievalResult:
        """Query documents within tenant scope.

        Args:
            query: Query string.
            top_k: Number of results to return.
            tenant_id: Target tenant. Uses context if None.

        Returns:
            TenantRetrievalResult with documents and timing.
        """
        target_tenant = tenant_id or self.tenant_context.tenant_id

        with Timer() as timer:
            # Embed query
            query_embedding = self._embedder.run(text=query)["embedding"]

            # Truncate if output_dimension specified
            output_dim = self.config.get("embedding", {}).get("output_dimension")
            if output_dim and query_embedding:
                query_embedding = query_embedding[:output_dim]

            # Perform vector search with tenant isolation
            class_name = self._get_class_name()

            result = (
                self._client.query.get(class_name, ["content", "tenant_id"])
                .with_tenant(target_tenant)  # This is key for Weaviate multi-tenancy
                .with_near_vector({"vector": query_embedding})
                .with_limit(top_k)
                .do()
            )

            # Extract documents from result
            documents = []
            scores = []

            if "data" in result and "Get" in result["data"]:
                class_results = result["data"]["Get"].get(class_name, [])
                for item in class_results:
                    doc = Document(
                        content=item.get("content", ""),
                        meta={"tenant_id": item.get("tenant_id", target_tenant)},
                    )
                    documents.append(doc)
                    # Placeholder for score - actual score would come from Weaviate
                    scores.append(1.0)

        query_results = []
        for i, doc in enumerate(documents):
            query_result = TenantQueryResult(
                document=doc,
                relevance_score=scores[i] if i < len(scores) else 1.0,
                rank=i + 1,
                tenant_id=target_tenant,
            )
            query_results.append(query_result)

        metrics = self._create_timing_metrics(timer.elapsed_ms)

        result = TenantRetrievalResult(
            tenant_id=target_tenant,
            query=query,
            documents=documents,
            scores=scores,
            timing=metrics,
        )

        logger.info(
            "Queried tenant %s, retrieved %d documents in %.1fms",
            target_tenant,
            len(documents),
            timer.elapsed_ms,
        )

        return result

    def rag(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
    ) -> TenantRAGResult:
        """Perform RAG (Retrieval-Augmented Generation) within tenant scope.

        Args:
            query: Query string.
            top_k: Number of documents to retrieve.
            tenant_id: Target tenant. Uses context if None.

        Returns:
            TenantRAGResult with retrieved documents and generated response.

        Raises:
            ValueError: If RAG pipeline is not configured/enabled.
        """
        if not self._rag_pipeline:
            raise ValueError("RAG pipeline not enabled/configured")

        target_tenant = tenant_id or self.tenant_context.tenant_id

        with Timer() as timer:
            retrieval_result = self.query(query, top_k, target_tenant)
            retrieved_docs = retrieval_result.documents

            rag_result = self._rag_pipeline.run(
                {
                    "prompt_builder": {
                        "query": query,
                        "documents": retrieved_docs,
                    }
                }
            )

            generated_response = rag_result["generator"]["replies"][0]

        metrics = self._create_timing_metrics(timer.elapsed_ms)

        result = TenantRAGResult(
            tenant_id=target_tenant,
            query=query,
            retrieved_documents=retrieved_docs,
            generated_response=generated_response,
            timing=metrics,
            retrieval_scores=retrieval_result.scores,
        )

        logger.info(
            "RAG query for tenant %s completed in %.1fms",
            target_tenant,
            timer.elapsed_ms,
        )

        return result

    def _create_timing_metrics(
        self,
        total_ms: float,
    ) -> MultitenancyTimingMetrics:
        """Create timing metrics for search operation.

        Args:
            total_ms: Total operation time in milliseconds.

        Returns:
            Timing metrics object.
        """
        return MultitenancyTimingMetrics(
            tenant_resolution_ms=0.0,
            index_operation_ms=0.0,
            retrieval_ms=total_ms,
            total_ms=total_ms,
            tenant_id=self.tenant_context.tenant_id,
            num_documents=0,
        )
