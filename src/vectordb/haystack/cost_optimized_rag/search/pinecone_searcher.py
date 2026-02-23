"""Pinecone search and RAG pipeline for cost-optimized RAG.

Fully managed vector search with Pinecone's serverless architecture.
Optimized for teams wanting zero operational overhead with pay-per-query
pricing. Best for applications prioritizing reliability over cost optimization.

Cost Architecture:

    Managed Service Pricing:
        - Storage: ~$0.10 per GB/month
        - Queries: ~$0.001 per 1000 queries (varies by pod type)
        - No infrastructure management costs
        - Auto-scaling included in query pricing

    Query Cost Optimization:
        - Over-fetch strategy (2x top_k) reduces reranking calls
        - Configurable top_k controls query volume
        - Metadata filtering reduces scanned vectors
        - Namespace isolation for multi-tenant cost attribution

    Embedding Strategy:
        - Local sentence-transformers: $0 per query
        - Alternative: Pinecone inference (additional cost)
        - Query batching reduces per-query overhead

Token Usage Optimization:

    RAG Generation:
        - max_tokens limits response length (cost control)
        - temperature affects output length variance
        - Document selection impacts context window usage
        - Concise prompt templates minimize input tokens

Performance vs Cost Trade-offs:

    Pod Type Selection:
        - p1: Balanced performance/cost (default)
        - p2: Higher throughput, higher cost
        - s1: Standard performance, lowest cost

    Index Configuration:
        - Metadata indexes: Enable filtering (small storage cost)
        - No metadata indexes: Lower storage, slower filtered queries

When to Use Pinecone:
    - Want zero operational overhead
    - Need reliable, managed service
    - Cost predictability important
    - Don't want to manage vector DB infrastructure

When to Consider Self-Hosted:
    - High query volume (>1M/month) - self-hosted cheaper
    - Complex custom requirements
    - Strict data residency requirements
    - Budget for dedicated infrastructure team
"""

from pathlib import Path
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import Secret
from pinecone import Pinecone

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    format_search_results,
)
from vectordb.haystack.cost_optimized_rag.utils.prompt_templates import (
    RAG_ANSWER_TEMPLATE,
)


class PineconeSearchPipeline:
    """Pinecone search pipeline with optional reranking and RAG generation.

    Implements cost-efficient search with over-fetch strategy for reranking.
    Uses local embeddings to minimize query costs.

    Cost Architecture:
        - Search: Pinecone query costs (~$0.001/1K queries)
        - Embedding: $0 (local sentence-transformers)
        - Reranking: Optional (local model, zero cost)
        - Generation: LLM API costs (configurable max_tokens)

    Performance Characteristics:
        - Query latency: 50-150ms (Pinecone managed)
        - Throughput: Scales with Pinecone pods
        - Reranking: Adds 20-100ms for cross-encoder

    Over-fetch Strategy:
        - Retrieves 2x top_k when reranking enabled
        - Improves reranker input quality
        - Cost: 2x Pinecone queries but better results

    Token Optimization:
        - max_tokens controls generation cost
        - Temperature affects output variance
        - Concise templates minimize prompt tokens

    Example:
        >>> pipeline = PineconeSearchPipeline("config.yaml")
        >>> results = pipeline.search("query", top_k=10)
        >>> rag_result = pipeline.search_with_rag("query", top_k=5)
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML with Pinecone API key, embedding
                settings, reranker config, and generator parameters.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        self._connect()
        self._init_embedder()
        self._init_ranker()
        self._init_rag_pipeline()

    def _connect(self) -> None:
        """Connect to Pinecone index."""
        if self.config.pinecone is None:
            msg = "Pinecone configuration is missing"
            raise ValueError(msg)

        self.pc = Pinecone(api_key=self.config.pinecone.api_key)
        self.index = self.pc.Index(self.config.collection.name)
        self.logger.info("Connected to Pinecone index: %s", self.config.collection.name)

    def _init_embedder(self) -> None:
        """Initialize local text embedder (zero per-query cost)."""
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
        """Initialize RAG pipeline with configurable generation parameters.

        Generation cost controlled via:
            - max_tokens: Hard limit on output length
            - temperature: Affects quality/creativity trade-off
        """
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
        """Execute vector search with optional reranking.

        Uses over-fetch strategy (2x top_k) when reranking enabled
        to improve reranker input quality.

        Args:
            query: Query text to search.
            top_k: Number of results to return. If reranking enabled,
                fetches 2x for reranker input pool.

        Returns:
            List of search results with id, content, score, metadata.
        """
        top_k = top_k or self.config.search.top_k

        # Local embedding - zero API cost
        embed_result = self.embedder.run(text=query)
        query_embedding = embed_result["embedding"]

        # Over-fetch for reranking quality
        search_top_k = top_k * 2 if self.ranker else top_k
        results = self.index.query(
            vector=query_embedding,
            top_k=search_top_k,
            namespace=self.config.collection.name,
            include_metadata=True,
        )

        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            content = metadata.pop("content", "")
            documents.append(
                Document(
                    id=match.get("id", ""),
                    content=content,
                    meta={**metadata},
                    score=match.get("score", 0.0),
                )
            )

        # Rerank if enabled (local model, zero cost)
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
            1. Vector search (Pinecone query cost)
            2. Optional reranking (local, free)
            3. LLM generation (API cost based on tokens)

        Args:
            query: Query text.
            top_k: Number of documents to retrieve for context.

        Returns:
            Dict with 'documents' (retrieved) and 'answer' (generated).
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
