"""Milvus search and RAG pipeline for cost-optimized RAG.

Distributed vector search with Milvus's partitioned architecture.
Optimized for high-throughput applications with horizontal scalability
and hybrid search capabilities (vector + scalar filtering).

Cost Architecture:

    Self-Hosted Infrastructure:
        - Index nodes: CPU for ingestion
        - Query nodes: RAM for vector search
        - Storage: Raw vectors + indexes
        - Coordination: Minimal overhead

    Managed Service (Zilliz):
        - CU-based pricing (~$0.10/hour per CU)
        - 1 CU = 1 vCPU + 4GB RAM
        - Auto-scaling based on load
        - Predictable costs at scale

    Query Cost Factors:
        - nprobe: Higher = better recall, slower
        - Top_k: Larger = more data transfer
        - Filters: Good selectivity = faster
        - Partitions: Reduce scan scope

Token Usage Optimization:

    RAG Context Assembly:
        - top_k controls document count
        - Partition filtering reduces retrieved docs
        - Metadata filtering for relevance

    Generation Cost Control:
        - max_tokens: Hard output limit
        - temperature: Quality vs speed
        - Model selection (Groq cheaper than GPT-4)

Performance vs Cost Trade-offs:

    Index Configuration:
        - IVF_FLAT: Balanced (default)
        - IVF_SQ8: 75% memory savings
        - IVF_PQ: Higher compression, lower recall
        - HNSW: Fastest, most memory

    Query Parameters:
        - nprobe=10: Fast, good recall
        - nprobe=100: Slower, better recall
        - Filter selectivity matters most

When to Use Milvus:
    - Billion-scale collections
    - Need horizontal scaling
    - Hybrid search requirements
    - Multi-tenancy with partitions

When to Consider Alternatives:
    - Small collections (<1M) - simpler solutions
    - No partitioning needs - Qdrant simpler
    - Zero ops - Pinecone managed
"""

from pathlib import Path
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import Secret
from pymilvus import Collection, connections

from vectordb.haystack.cost_optimized_rag.base.config import load_config
from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    format_search_results,
)
from vectordb.haystack.cost_optimized_rag.utils.prompt_templates import (
    RAG_ANSWER_TEMPLATE,
)


class MilvusSearchPipeline:
    """Milvus search pipeline with partition support and hybrid filtering.

    Supports vector similarity search with optional scalar filtering
    and reranking. Designed for high-throughput distributed deployments.

    Cost Architecture:
        - Search: Self-hosted (infra) or managed (CU-based)
        - Embedding: $0 (local sentence-transformers)
        - Reranking: Local model, zero cost
        - Generation: Token-based LLM costs

    Performance Characteristics:
        - Query latency: 10-50ms (IVF_FLAT)
        - Throughput: 1000-5000 qps per query node
        - Scales horizontally with query nodes

    Search Strategy:
        - Vector similarity with IVF index
        - nprobe controls recall/speed trade-off
        - Output fields minimize data transfer

    Token Optimization:
        - max_tokens limits generation cost
        - top_k balances context quality/cost

    Example:
        >>> pipeline = MilvusSearchPipeline("config.yaml")
        >>> results = pipeline.search("query", top_k=10)
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML with Milvus connection,
                embedding, reranker, and generator settings.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        self._connect()
        self._init_embedder()
        self._init_ranker()
        self._init_rag_pipeline()

    def _connect(self) -> None:
        """Connect to Milvus and load collection."""
        if self.config.milvus is None:
            msg = "Milvus configuration is missing"
            raise ValueError(msg)

        connections.connect(
            alias="default",
            host=self.config.milvus.host,
            port=self.config.milvus.port,
        )
        self.collection = Collection(name=self.config.collection.name)
        self.collection.load()
        self.logger.info(
            "Connected to Milvus collection: %s", self.config.collection.name
        )

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
        """Initialize RAG generation with cost controls."""
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
        """Execute vector search with IVF index and optional reranking.

        Uses IVF_FLAT index with configurable nprobe for recall/speed
        trade-off. Over-fetches 2x when reranking enabled.

        Args:
            query: Query text.
            top_k: Number of results. 2x fetched if reranking.

        Returns:
            Search results with content, metadata, and L2 distances.
        """
        top_k = top_k or self.config.search.top_k

        # Local embedding
        embed_result = self.embedder.run(text=query)
        query_embedding = embed_result["embedding"]

        # IVF search with nprobe for recall tuning
        search_top_k = top_k * 2 if self.ranker else top_k
        search_result = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=search_top_k,
            output_fields=["content", "metadata"],
        )

        documents = []
        for hit in search_result[0]:
            documents.append(
                Document(
                    id=hit.id,
                    content=hit.entity.get("content", ""),
                    meta=hit.entity.get("metadata", {}),
                    score=float(hit.distance),
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
