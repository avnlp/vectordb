"""Chroma search and RAG pipeline for cost-optimized RAG.

Local vector search with Chroma's persistent storage.
Optimized for development, testing, and small-scale production where
operational simplicity and zero infrastructure costs are priorities.

Cost Architecture:

    Zero Infrastructure Costs:
        - Local SQLite + file storage
        - No cloud service fees
        - Runs on application server
        - Backup via simple file copy

    Resource Usage:
        - Memory: HNSW index loaded on query
        - Storage: ~200MB per 100k vectors (768-dim)
        - CPU: Embedding inference only

    Query Economics:
        - Unlimited queries (no metered pricing)
        - Latency acceptable for most use cases
        - Single-node limitation

Token Usage Optimization:

    Context Assembly:
        - top_k controls prompt size
        - Document content affects tokens
        - Metadata inclusion configurable

    Generation Efficiency:
        - max_tokens: Output limit
        - temperature: Response variance
        - Concise templates minimize input

Performance Characteristics:

    Query Latency:
        - 10-50ms typical for <1M vectors
        - Scales linearly with collection size
        - Memory-bound (index in RAM)

    Throughput:
        - 100-500 qps single node
        - CPU-bound by embedding
        - SQLite concurrency limits

When to Use Chroma:
    - Development and prototyping
    - Small production deployments
    - Zero infrastructure overhead
    - Cost-sensitive applications

When to Consider Alternatives:
    - Large scale (>10M vectors)
    - High availability requirements
    - Distributed query needs
    - Advanced hybrid search
"""

from pathlib import Path
from typing import Any

import chromadb
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


class ChromaSearchPipeline:
    """Chroma search pipeline with local persistent storage.

    Simplified vector search for cost-conscious deployments.
    No infrastructure costs beyond local storage.

    Cost Architecture:
        - Search: $0 (local processing)
        - Storage: Local disk only
        - Embedding: $0 (local model)
        - Generation: Token-based LLM costs

    Performance Characteristics:
        - Query latency: 10-50ms
        - Throughput: 100-500 qps
        - Memory: HNSW index on demand

    Search Strategy:
        - L2 distance with score conversion
        - Over-fetch for reranking
        - Simple metadata retrieval

    Token Optimization:
        - max_tokens limits cost
        - top_k balances quality

    Example:
        >>> pipeline = ChromaSearchPipeline("config.yaml")
        >>> results = pipeline.search("query", top_k=10)
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from config file.

        Args:
            config_path: Path to YAML with Chroma path, embedding,
                reranker, and generator settings.
        """
        self.config = load_config(str(config_path))
        self.logger = create_logger(self.config)

        self._connect()
        self._init_embedder()
        self._init_ranker()
        self._init_rag_pipeline()

    def _connect(self) -> None:
        """Initialize Chroma persistent client."""
        if self.config.chroma is None:
            msg = "Chroma configuration is missing"
            raise ValueError(msg)

        self.client = chromadb.PersistentClient(path=self.config.chroma.path)
        self.collection = self.client.get_collection(name=self.config.collection.name)
        self.logger.info(
            "Connected to Chroma collection: %s", self.config.collection.name
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
        """Execute vector search with optional reranking.

        Queries Chroma with L2 distance and converts to similarity score.
        Over-fetches 2x when reranking enabled.

        Args:
            query: Query text.
            top_k: Number of results. 2x fetched if reranking.

        Returns:
            Search results with converted similarity scores.
        """
        top_k = top_k or self.config.search.top_k

        # Local embedding
        embed_result = self.embedder.run(text=query)
        query_embedding = embed_result["embedding"]

        # Over-fetch for reranking quality
        search_top_k = top_k * 2 if self.ranker else top_k
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_top_k,
            include=["metadatas", "documents", "distances"],
        )

        documents = []
        ids_list = results["ids"][0]
        distances_list = results["distances"][0]
        documents_list = results["documents"][0]
        metadatas_list = results["metadatas"][0]

        for i in range(len(ids_list)):
            # Convert L2 distance to similarity score
            score = 1.0 / (1.0 + distances_list[i])
            documents.append(
                Document(
                    id=ids_list[i],
                    content=documents_list[i],
                    meta=metadatas_list[i],
                    score=score,
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
