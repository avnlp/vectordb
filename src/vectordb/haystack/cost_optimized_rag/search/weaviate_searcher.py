"""Weaviate search and RAG pipeline for cost-optimized RAG.

GraphQL-based vector search with Weaviate's modular architecture.
Optimized for applications needing flexible query capabilities beyond
simple vector similarity, with optional hybrid search support.

Cost Architecture:

    Self-Hosted Deployment:
        - Infrastructure: Compute + storage costs
        - No per-query fees
        - Memory-bound (vectors must fit in RAM)
        - Horizontal scaling available for query nodes

    Weaviate Cloud:
        - ~$0.50 per GB/month storage
        - Query costs based on SLA tier
        - Managed backups and monitoring included

    Query Cost Factors:
        - Vector search: Fast HNSW lookups
        - Hybrid search: Additional BM25 cost
        - GraphQL complexity affects query time
        - Filters reduce scanned vectors

Token Usage Optimization:

    RAG Context Window:
        - top_k controls document count in prompt
        - Document length affects token usage
        - Truncation strategies for long docs
        - Metadata inclusion configurable

    Generation Parameters:
        - max_tokens: Hard output limit
        - temperature: Quality/creativity balance
        - Concise prompt templates

Performance vs Cost Trade-offs:

    Index Types:
        - HNSW: Fast queries, more RAM
        - Flat: Slower, less RAM
        - Dynamic: Balances both

    Query Patterns:
        - Pure vector: Fastest, lowest cost
        - Hybrid: Better recall, higher cost
        - Filtered: Faster with good selectivity

When to Use Weaviate:
    - Need GraphQL query interface
    - Want modular capabilities (reranking, QA)
    - Hybrid search requirements
    - Prefer self-hosted with flexibility

When to Consider Alternatives:
    - Simple vector-only search - Qdrant/Chroma simpler
    - Zero ops requirement - Pinecone managed
    - Ultra-low latency - specialized solutions
"""

import json
from pathlib import Path
from typing import Any

import weaviate
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import Secret

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    format_search_results,
)
from vectordb.haystack.cost_optimized_rag.utils.prompt_templates import (
    RAG_ANSWER_TEMPLATE,
)


class WeaviateSearchPipeline:
    """Weaviate search pipeline with GraphQL interface and RAG support.

    Supports vector search with optional reranking and LLM generation.
    Uses GraphQL for flexible result retrieval.

    Cost Architecture:
        - Search: Self-hosted (infra cost) or Cloud ($/GB)
        - Embedding: $0 (local sentence-transformers)
        - Reranking: Local model, zero cost
        - Generation: LLM API costs (token-based)

    Performance Characteristics:
        - Query latency: 20-100ms (depends on index)
        - GraphQL overhead: ~5-10ms parsing
        - Reranking: +20-80ms for cross-encoder

    Query Strategy:
        - near_vector for pure similarity
        - bm25 for keyword fallback
        - hybrid for combined approach

    Token Optimization:
        - max_tokens limits generation cost
        - Document count (top_k) affects context size
        - Concise templates minimize prompt length

    Example:
        >>> pipeline = WeaviateSearchPipeline("config.yaml")
        >>> results = pipeline.search("query", top_k=10)
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML with Weaviate connection,
                embedding model, reranker, and generator settings.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        self._connect()
        self._init_embedder()
        self._init_ranker()
        self._init_rag_pipeline()

    def _connect(self) -> None:
        """Connect to Weaviate with configured authentication."""
        if self.config.weaviate is None:
            msg = "Weaviate configuration is missing"
            raise ValueError(msg)

        self.client = weaviate.Client(
            url=f"http://{self.config.weaviate.host}:{self.config.weaviate.port}",
            auth_client_secret=weaviate.auth.AuthApiKey(self.config.weaviate.api_key)
            if self.config.weaviate.api_key
            else None,
        )
        self.logger.info("Connected to Weaviate")

    def _init_embedder(self) -> None:
        """Initialize local text embedder."""
        self.embedder = SentenceTransformersTextEmbedder(
            model=self.config.embeddings.model,
        )
        self.embedder.warm_up()

    def _init_ranker(self) -> None:
        """Initialize optional reranker for result refinement."""
        self.ranker = None
        if self.config.search.reranking_enabled:
            self.ranker = SentenceTransformersSimilarityRanker(
                model=self.config.reranker.model,
                top_k=self.config.reranker.top_k,
            )
            self.ranker.warm_up()
            self.logger.info("Initialized ranker: %s", self.config.reranker.model)

    def _init_rag_pipeline(self) -> None:
        """Initialize RAG generation pipeline with token cost controls."""
        self.rag_pipeline = None

        if not self.config.generator.enabled:
            self.logger.info("RAG generator disabled in config")
            return

        api_key = self.config.generator.api_key
        if not api_key:
            self.logger.info("No generator API key configured, RAG disabled")
            return

        prompt_builder = PromptBuilder(template=RAG_ANSWER_TEMPLATE)
        generator = OpenAIGenerator(
            api_key=Secret.from_token(api_key),
            api_base_url=self.config.generator.api_base_url,
            model=self.config.generator.model,
            generation_kwargs={
                "temperature": self.config.generator.temperature,
                "max_tokens": self.config.generator.max_tokens,
            },
        )

        self.rag_pipeline = Pipeline()
        self.rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.rag_pipeline.add_component("llm", generator)
        self.rag_pipeline.connect("prompt_builder", "llm")
        self.logger.info(
            "Initialized RAG pipeline with model: %s", self.config.generator.model
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute vector search via GraphQL with optional reranking.

        Uses near_vector GraphQL query with configurable top_k.
        Over-fetches 2x when reranking enabled.

        Args:
            query: Query text.
            top_k: Number of results. 2x fetched if reranking.

        Returns:
            Search results with content, metadata, and scores.
        """
        top_k = top_k or self.config.search.top_k

        # Local embedding
        embed_result = self.embedder.run(text=query)
        query_embedding = embed_result["embedding"]

        # GraphQL near_vector query
        class_name = self.config.collection.name
        search_top_k = top_k * 2 if self.ranker else top_k

        results = (
            self.client.query.get(
                class_name,
                ["content", "metadata", "_id"],
            )
            .with_near_vector({"vector": query_embedding})
            .with_limit(search_top_k)
            .do()
        )

        documents = []
        for obj in results.get("data", {}).get("Get", {}).get(class_name, []):
            metadata_str = obj.get("metadata", "{}")
            try:
                metadata = json.loads(metadata_str)
            except Exception:
                metadata = {"raw": metadata_str}

            documents.append(
                Document(
                    id=obj.get("_id", ""),
                    content=obj.get("content", ""),
                    meta=metadata,
                    score=1.0,
                )
            )

        # Rerank if enabled
        if self.ranker and documents:
            rank_result = self.ranker.run(query=query, documents=documents)
            documents = rank_result["documents"][:top_k]

        return format_search_results(documents)

    def search_with_rag(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Execute search and generate RAG answer.

        Cost components:
            1. Vector search (Weaviate query)
            2. Optional reranking (local)
            3. LLM generation (token-based cost)

        Args:
            query: Query text.
            top_k: Documents to retrieve.

        Returns:
            Dict with documents and generated answer.
        """
        results = self.search(query, top_k)
        documents = [
            Document(id=r["id"], content=r["content"], meta=r["metadata"])
            for r in results
        ]

        answer = None
        if self.rag_pipeline and documents:
            rag_result = self.rag_pipeline.run(
                {
                    "prompt_builder": {"documents": documents, "query": query},
                }
            )
            replies = rag_result.get("llm", {}).get("replies", [])
            answer = replies[0] if replies else None

        return {
            "documents": results,
            "answer": answer,
        }
