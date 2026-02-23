"""Qdrant search and RAG pipeline for cost-optimized RAG.

High-performance vector search with Qdrant's HNSW indexing and payload
filtering. Optimized for hybrid search scenarios requiring fast metadata
filtering alongside vector similarity.

Cost Architecture:

    Self-Hosted Deployment:
        - Storage: Vectors + HNSW index + payload data
        - Compute: Query nodes for search
        - Network: gRPC overhead minimal
        - No per-query fees

    Qdrant Cloud:
        - ~$0.05 per GB/month storage
        - Query costs based on CPU
        - Free tier available
        - Managed backups included

    Query Cost Factors:
        - Payload filtering: Pre-filter reduces vector scan
        - HNSW ef parameter: Higher = better recall, slower
        - Top_k: Affects result transfer size
        - Batch queries: Amortize connection overhead

Token Usage Optimization:

    Context Control:
        - top_k limits document count
        - Payload filtering for relevance
        - Metadata selection reduces context size

    Generation Efficiency:
        - max_tokens: Hard output limit
        - temperature: Response quality
        - Concise prompt templates

Performance vs Cost Trade-offs:

    HNSW Tuning:
        - ef=64: Fast queries, good recall
        - ef=256: Better recall, slower
        - ef=512: Best recall, 4x slower

    Payload Indexes:
        - Add storage overhead (~10%)
        - Enable fast pre-filtering
        - Essential for hybrid search

When to Use Qdrant:
    - Need hybrid search (vector + filter)
    - Want efficient payload filtering
    - Real-time index updates
    - Prefer simpler ops than Milvus

When to Consider Alternatives:
    - Massive scale (billions) - Milvus
    - Ultra-simple needs - Chroma
    - Zero ops - Pinecone
"""

from pathlib import Path
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import Secret
from qdrant_client import QdrantClient

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    format_search_results,
)
from vectordb.haystack.cost_optimized_rag.utils.prompt_templates import (
    RAG_ANSWER_TEMPLATE,
)


class QdrantSearchPipeline:
    """Qdrant search pipeline with payload filtering and HNSW search.

    Optimized for hybrid search combining vector similarity with
    metadata filtering. Uses HNSW index for fast ANN queries.

    Cost Architecture:
        - Search: Self-hosted or managed cloud
        - Embedding: $0 (local sentence-transformers)
        - Reranking: Local model, zero cost
        - Generation: Token-based LLM costs

    Performance Characteristics:
        - Query latency: 5-20ms (HNSW)
        - Payload filtering: Reduces scan scope
        - Throughput: 1000-5000 qps per node

    Search Strategy:
        - HNSW with configurable ef parameter
        - Payload indexes enable fast filtering
        - Over-fetch for reranking quality

    Token Optimization:
        - max_tokens controls generation cost
        - top_k balances context quality

    Example:
        >>> pipeline = QdrantSearchPipeline("config.yaml")
        >>> results = pipeline.search("query", top_k=10)
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML with Qdrant connection,
                embedding, reranker, and generator settings.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        self._connect()
        self._init_embedder()
        self._init_ranker()
        self._init_rag_pipeline()

    def _connect(self) -> None:
        """Connect to Qdrant with TLS and authentication."""
        if self.config.qdrant is None:
            msg = "Qdrant configuration is missing"
            raise ValueError(msg)

        self.client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            api_key=self.config.qdrant.api_key or None,
            https=self.config.qdrant.https,
        )
        self.logger.info("Connected to Qdrant")

    def _init_embedder(self) -> None:
        """Initialize local text embedder."""
        self.embedder = SentenceTransformersTextEmbedder(
            model=self.config.embeddings.model,
        )
        self.embedder.warm_up()

    def _init_ranker(self) -> None:
        """Initialize optional reranker."""
        self.ranker = None
        if self.config.search.reranking_enabled:
            self.ranker = SentenceTransformersSimilarityRanker(
                model=self.config.reranker.model,
                top_k=self.config.reranker.top_k,
            )
            self.ranker.warm_up()
            self.logger.info("Initialized ranker: %s", self.config.reranker.model)

    def _init_rag_pipeline(self) -> None:
        """Initialize RAG generation pipeline."""
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
        """Execute HNSW vector search with optional reranking.

        Uses Qdrant's search API with configurable top_k.
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

        # HNSW search with over-fetch for reranking
        collection_name = self.config.collection.name
        search_top_k = top_k * 2 if self.ranker else top_k

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=search_top_k,
        )

        documents = []
        for scored_point in results:
            content = scored_point.payload.get("content", "")
            metadata = {k: v for k, v in scored_point.payload.items() if k != "content"}
            documents.append(
                Document(
                    id=str(scored_point.id),
                    content=content,
                    meta=metadata,
                    score=scored_point.score,
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
