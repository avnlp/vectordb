"""Weaviate query enhancement search pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.components import QueryEnhancer
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    ResultMerger,
)


logger = logging.getLogger(__name__)


class WeaviateQueryEnhancementSearchPipeline:
    """Weaviate query enhancement search pipeline (LangChain).

    Enhances query with multiple perspectives, performs parallel searches,
    and fuses results using RRF.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name")

        llm = RAGHelper.create_llm(self.config)
        if llm is None:
            from langchain_groq import ChatGroq

            llm = ChatGroq(model="llama-3.3-70b-versatile")

        self.query_enhancer = QueryEnhancer(llm)

        # Optional RAG
        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Weaviate query enhancement search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        mode: str = "multi_query",
    ) -> dict[str, Any]:
        """Execute query enhancement search.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            mode: Query enhancement mode ('multi_query', 'hyde', 'step_back').

        Returns:
            Dict with 'documents', 'query', 'enhanced_queries', and optional
            'answer' keys.
        """
        logger.info("Starting query enhancement search (mode=%s)", mode)

        enhanced_queries = self.query_enhancer.generate_queries(query, mode=mode)
        logger.info("Generated %d enhanced queries", len(enhanced_queries))

        # Perform parallel searches
        all_results = []
        for enhanced_query in enhanced_queries:
            query_embedding = EmbedderHelper.embed_query(self.embedder, enhanced_query)
            documents = self.db.query(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
                collection_name=self.collection_name,
            )
            all_results.append(documents)
            logger.info(
                "Retrieved %d documents for query: %s",
                len(documents),
                enhanced_query[:50],
            )

        # Fuse results using RRF
        fused_documents = ResultMerger.reciprocal_rank_fusion(all_results, k=60)
        fused_documents = fused_documents[:top_k]
        logger.info("Fused results: %d documents", len(fused_documents))

        result = {
            "documents": fused_documents,
            "query": query,
            "enhanced_queries": enhanced_queries,
        }

        # Optional RAG generation
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, fused_documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
