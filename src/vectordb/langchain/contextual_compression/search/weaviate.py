"""Weaviate contextual compression search pipeline (LangChain).

This module provides a search pipeline for Weaviate vector database with
contextual compression support. The pipeline retrieves documents from Weaviate,
applies compression, and optionally generates answers.

Weaviate-Specific Features:
    - Schema-based retrieval: Searches within a defined class
    - GraphQL queries: Uses GraphQL for flexible querying
    - Hybrid search: Can combine vector and keyword search
    - Rich filtering: Supports complex filter expressions

Contextual Compression in Weaviate:
    The pipeline follows the standard two-stage retrieval pattern:
    1. Over-fetch: Retrieve 2*top_k documents from the class
    2. Compress: Apply reranking or LLM extraction to reduce to top_k

    Weaviate's hybrid search capability can be used in the retrieval phase
    to improve the quality of the candidate set before compression.

Configuration Schema:
    Required sections:
    - weaviate: URL, API key, class/collection settings
    - embedding: Query embedding configuration
    - compression: Mode and model settings
    - rag (optional): LLM for answer generation
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.components import ContextCompressor
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


# Module-level logger for search operations
logger = logging.getLogger(__name__)


class WeaviateContextualCompressionSearchPipeline:
    """Search pipeline for Weaviate with contextual compression.

    This pipeline implements contextual compression-based retrieval for Weaviate.
    It retrieves documents from a class, compresses them using reranking
    or LLM extraction, and optionally generates answers.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for query vectorization
        db: WeaviateVectorDB instance for document retrieval
        collection_name: Name of the Weaviate class to search
        compressor: ContextCompressor instance for document compression
        llm: Optional LLM for answer generation

    Design Decisions:
        - Class-scoped: All operations target a specific Weaviate class
        - Schema-aware: Respects Weaviate's schema definitions
        - Flexible querying: Supports both vector and hybrid search modes
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the search pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'weaviate', 'embedding', and
                optionally 'compression' and 'rag' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name")

        compression_config = self.config.get("compression", {})
        mode = compression_config.get("mode", "reranking")

        if mode == "reranking":
            reranker = RerankerHelper.create_reranker(self.config)
            self.compressor = ContextCompressor(mode="reranking", reranker=reranker)
        else:  # llm_extraction
            llm_extraction_config = compression_config.get("llm_extraction", {})
            if not llm_extraction_config:
                raise ValueError(
                    "llm_extraction config section is required for 'llm_extraction' mode."
                )

            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=llm_extraction_config.get("model"),
                api_key=llm_extraction_config.get("api_key"),
            )
            self.compressor = ContextCompressor(mode="llm_extraction", llm=llm)

        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Weaviate contextual compression search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute contextual compression search.

        Args:
            query: The search query text.
            top_k: Number of documents to return after compression.
            filters: Optional filters for Weaviate query.

        Returns:
            Dictionary containing:
                - 'documents': List of compressed Document objects
                - 'query': The original query string
                - 'answer': Generated answer (if LLM configured)
        """
        logger.info("Starting contextual compression search")

        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Over-fetch from Weaviate class to enable compression
        retrieved_documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d documents from Weaviate", len(retrieved_documents))

        # Apply contextual compression: reranking or LLM extraction
        compressed_documents = self.compressor.compress(
            query=query,
            documents=retrieved_documents,
            top_k=top_k,
        )
        logger.info("Compressed to %d documents", len(compressed_documents))

        result = {
            "documents": compressed_documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, compressed_documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
