"""Base class for LangChain query enhancement search pipelines."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from vectordb.langchain.components import QueryEnhancer
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    ResultMerger,
)


logger = logging.getLogger(__name__)


class BaseQueryEnhancementSearchPipeline(ABC):
    """Abstract base for query enhancement search pipelines (LangChain).

    Handles query enhancement, parallel search, RRF fusion, and optional RAG.
    Subclasses implement `_db_key` and `_initialize_db` for database-specific setup,
    and `_perform_search` for database-specific query execution.
    """

    @property
    @abstractmethod
    def _db_key(self) -> str:
        """Config section key for the database (e.g. 'chroma', 'milvus')."""

    @abstractmethod
    def _initialize_db(self, db_config: dict[str, Any]) -> Any:
        """Initialize and return the vector database client."""

    @abstractmethod
    def _perform_search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[Document]:
        """Execute a single search against the vector database."""

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, self._db_key)

        self.embedder = EmbedderHelper.create_embedder(self.config)
        self.db = self._initialize_db(self.config[self._db_key])

        llm = RAGHelper.create_llm(self.config)
        if llm is None:
            from langchain_groq import ChatGroq

            llm = ChatGroq(model="llama-3.3-70b-versatile")

        self.query_enhancer = QueryEnhancer(llm)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized %s (LangChain)", self.__class__.__name__)

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
            top_k: Number of results to return after fusion.
            filters: Optional metadata filters.
            mode: Query enhancement mode ('multi_query', 'hyde', 'step_back').

        Returns:
            Dict with 'documents', 'query', 'enhanced_queries', and optional 'answer'.
        """
        logger.info("Starting query enhancement search (mode=%s)", mode)

        enhanced_queries = self.query_enhancer.generate_queries(query, mode=mode)
        logger.info("Generated %d enhanced queries", len(enhanced_queries))

        all_results: list[list[Document]] = []
        for enhanced_query in enhanced_queries:
            query_embedding = EmbedderHelper.embed_query(self.embedder, enhanced_query)
            documents = self._perform_search(query_embedding, top_k, filters)
            all_results.append(documents)
            logger.info(
                "Retrieved %d documents for query: %s",
                len(documents),
                enhanced_query[:50],
            )

        rrf_k = self.config.get("query_enhancement", {}).get("rrf_k", 60)
        fused_documents = ResultMerger.reciprocal_rank_fusion(all_results, k=rrf_k)
        fused_documents = fused_documents[:top_k]
        logger.info("Fused results: %d documents", len(fused_documents))

        result = {
            "documents": fused_documents,
            "query": query,
            "enhanced_queries": enhanced_queries,
        }

        if self.llm is not None:
            result["answer"] = RAGHelper.generate(self.llm, query, fused_documents)
            logger.info("Generated RAG answer")

        return result
