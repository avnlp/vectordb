"""Milvus contextual compression search pipeline (LangChain).

This module provides a search pipeline for Milvus vector database with
contextual compression support. The pipeline retrieves documents from Milvus,
applies compression, and optionally generates answers.

Milvus-Specific Features:
    - Collection-based retrieval: Searches within a named collection
    - Partition support: Can filter by partition for performance
    - Rich filtering: Supports complex metadata filter expressions
    - Scalable: Designed for high-throughput production workloads

Contextual Compression in Milvus:
    The pipeline follows the standard two-stage retrieval pattern:
    1. Over-fetch: Retrieve 2*top_k documents from the collection
    2. Compress: Apply reranking or LLM extraction to reduce to top_k

    Milvus's efficient ANN search makes over-fetching cost-effective,
    allowing the compression step to work with a diverse candidate set.

Configuration Schema:
    Required sections:
    - milvus: Host, port, and collection settings
    - embedding: Query embedding configuration
    - compression: Mode (reranking/llm_extraction) and model settings
    - rag (optional): LLM for answer generation
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.components import ContextCompressor
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


# Module-level logger for search operations
logger = logging.getLogger(__name__)


class MilvusContextualCompressionSearchPipeline:
    """Search pipeline for Milvus with contextual compression.

    This pipeline implements contextual compression-based retrieval for Milvus.
    It retrieves documents from a collection, compresses them using reranking
    or LLM extraction, and optionally generates answers.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for query vectorization
        db: MilvusVectorDB instance for document retrieval
        collection_name: Name of the Milvus collection to search
        compressor: ContextCompressor instance for document compression
        llm: Optional LLM for answer generation

    Design Decisions:
        - Collection-scoped: All operations target a specific collection
        - Efficient retrieval: Leverages Milvus's optimized ANN search
        - Configurable compression: Supports both fast and thorough modes
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the search pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'milvus', 'embedding', and
                optionally 'compression' and 'rag' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host"),
            port=milvus_config.get("port"),
            db_name=milvus_config.get("db_name"),
        )

        self.collection_name = milvus_config.get("collection_name")

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
            "Initialized Milvus contextual compression search pipeline (LangChain)"
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
            filters: Optional metadata filters for Milvus query.

        Returns:
            Dictionary containing:
                - 'documents': List of compressed Document objects
                - 'query': The original query string
                - 'answer': Generated answer (if LLM configured)
        """
        logger.info("Starting contextual compression search")

        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Over-fetch from Milvus collection to enable compression
        retrieved_documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d documents from Milvus", len(retrieved_documents))

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
