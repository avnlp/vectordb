"""Pinecone contextual compression search pipeline (LangChain).

This module provides a search pipeline for Pinecone vector database with
contextual compression support. The pipeline retrieves documents from Pinecone,
applies compression to filter/summarize content, and optionally generates answers.

Pinecone-Specific Features:
    - Namespace filtering: Retrieves documents from a specific namespace
    - Metadata filtering: Supports rich metadata filters during query
    - Cloud-based: All operations via API, no local storage
    - Hybrid search: Supports sparse-dense retrieval (if configured)

Contextual Compression in Pinecone:
    The pipeline follows the standard two-stage approach:
    1. Over-fetch: Retrieve 2*top_k documents from the specified namespace
    2. Compress: Apply reranking or LLM extraction to reduce to top_k

    Namespace isolation ensures compression only affects documents within
    the specified namespace, supporting multi-tenant use cases.

Configuration Schema:
    Required sections:
    - pinecone: API key, index name, namespace
    - embedding: Query embedding configuration
    - compression: Mode and model settings
    - rag (optional): LLM for answer generation
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.components import ContextCompressor
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


# Module-level logger for search operations
logger = logging.getLogger(__name__)


class PineconeContextualCompressionSearchPipeline:
    """Search pipeline for Pinecone with contextual compression.

    This pipeline implements contextual compression-based retrieval for Pinecone.
    It retrieves documents from a specific namespace, compresses them using
    reranking or LLM extraction, and optionally generates answers.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for query vectorization
        db: PineconeVectorDB instance for document retrieval
        index_name: Name of the Pinecone index
        namespace: Namespace within the index to search
        compressor: ContextCompressor instance for document compression
        llm: Optional LLM for answer generation

    Design Decisions:
        - Namespace-scoped: All operations target a specific namespace
        - Metadata support: Filters can be applied during retrieval
        - Efficient over-fetching: 2x retrieval allows better compression results
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the search pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'pinecone', 'embedding', and
                optionally 'compression' and 'rag' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        compression_config = self.config.get("compression", {})
        mode = compression_config.get("mode", "reranking")

        if mode == "reranking":
            reranker = RerankerHelper.create_reranker(self.config)
            self.compressor = ContextCompressor(mode="reranking", reranker=reranker)
        else:  # llm_extraction
            llm = RAGHelper.create_llm(self.config)
            if llm is None:
                from langchain_groq import ChatGroq

                llm = ChatGroq(model="llama-3.3-70b-versatile")
            self.compressor = ContextCompressor(mode="llm_extraction", llm=llm)

        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Pinecone contextual compression search pipeline (LangChain)"
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
            filters: Optional metadata filters for Pinecone query.

        Returns:
            Dictionary containing:
                - 'documents': List of compressed Document objects
                - 'query': The original query string
                - 'answer': Generated answer (if LLM configured)
        """
        logger.info("Starting contextual compression search")

        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Over-fetch documents from namespace for compression
        retrieved_documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filters=filters,
            namespace=self.namespace,
        )
        logger.info("Retrieved %d documents from Pinecone", len(retrieved_documents))

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
