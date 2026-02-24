"""Chroma multi-tenancy search pipeline with RAG support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from haystack import Pipeline

from vectordb.databases.chroma import ChromaVectorDB
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


class ChromaMultitenancySearchPipeline:
    """Chroma search pipeline with metadata-based tenant isolation.

    Provides both retrieval-only and RAG capabilities with tenant isolation.
    """

    TENANT_FIELD = "tenant_id"

    def __init__(
        self,
        config_or_path: dict[str, Any] | str | Path,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize Chroma search pipeline.

        Args:
            config_or_path: Config dict or path to YAML file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self.config = load_config(config_or_path)
        self.tenant_context = TenantContext.resolve(tenant_context, self.config)
        self._db: ChromaVectorDB | None = None
        self._embedder = None
        self._rag_pipeline: Pipeline | None = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Chroma and initialize components."""
        chroma_config = self.config.get("chroma", self.config.get("database", {}))

        self._db = ChromaVectorDB(
            collection_name=self._get_collection_name(),
            persist_dir=chroma_config.get("persist_dir", "./chroma_db"),
        )

        self._embedder = create_text_embedder(self.config)
        self._embedder.warm_up()

        if self.config.get("rag", {}).get("enabled", False):
            self._rag_pipeline = create_rag_pipeline(self.config)

    def _get_collection_name(self) -> str:
        """Get collection name from config."""
        # For Chroma, use a combination of the configured name and tenant for isolation
        base_name = self.config.get("collection", {}).get("name", "multitenancy")
        tenant_suffix = self.tenant_context.tenant_id.replace("-", "_").replace(
            ".", "_"
        )
        return f"{base_name}_{tenant_suffix}"

    def close(self) -> None:
        """Close Chroma connection."""
        # Chroma doesn't have a specific close method in most implementations
        pass

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

            # Query Chroma with tenant filter
            filters = {self.TENANT_FIELD: target_tenant}

            results = self._db.query(
                query_embedding=query_embedding,
                n_results=top_k,
                where=filters,
            )

            documents = self._db.query_to_documents(results)
            scores = [doc.score if doc.score is not None else 1.0 for doc in documents]

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
