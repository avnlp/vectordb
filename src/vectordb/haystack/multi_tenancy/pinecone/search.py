"""Pinecone multi-tenancy search pipeline with RAG support."""

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


class PineconeMultitenancySearchPipeline:
    """Pinecone search pipeline with namespace tenant isolation.

    Provides both retrieval-only and RAG capabilities with tenant isolation.
    """

    def __init__(
        self,
        config_or_path: dict[str, Any] | str | Path,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize Pinecone search pipeline.

        Args:
            config_or_path: Config dict or path to YAML file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).
        """
        self.config = load_config(config_or_path)
        self.tenant_context = TenantContext.resolve(tenant_context, self.config)
        self._index = None
        self._embedder = None
        self._rag_pipeline: Pipeline | None = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Pinecone and initialize components."""
        from pinecone import Pinecone

        pinecone_config = self.config.get("pinecone", self.config.get("database", {}))

        api_key = pinecone_config.get("api_key") or os.environ.get("PINECONE_API_KEY")

        self._pc = Pinecone(api_key=api_key)

        index_name = pinecone_config.get(
            "index", os.environ.get("PINECONE_INDEX", "multitenancy")
        )
        self._index = self._pc.Index(index_name)

        self._embedder = create_text_embedder(self.config)
        self._embedder.warm_up()

        if self.config.get("rag", {}).get("enabled", False):
            self._rag_pipeline = create_rag_pipeline(self.config)

    def _get_index_name(self) -> str:
        """Get index name from config."""
        pinecone_config = self.config.get("pinecone", self.config.get("database", {}))
        return pinecone_config.get(
            "index", os.environ.get("PINECONE_INDEX", "multitenancy")
        )

    def close(self) -> None:
        """Close Pinecone connection."""
        # Pinecone doesn't have a specific close method
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

            # Query Pinecone with tenant namespace
            result = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=target_tenant,  # This is key for Pinecone multi-tenancy
                include_metadata=True,
            )

            # Extract documents from result
            documents = []
            scores = []

            for match in result.matches:
                content = match.metadata.get("content", "") if match.metadata else ""
                doc = Document(content=content, meta=match.metadata or {})
                documents.append(doc)
                scores.append(match.score if hasattr(match, "score") else 1.0)

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
