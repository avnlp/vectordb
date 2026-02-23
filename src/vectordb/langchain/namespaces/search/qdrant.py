"""Qdrant namespace search pipeline (LangChain).

This module provides a search pipeline for Qdrant vector database with
namespace support using payload-based filtering.

Example:
    >>> from vectordb.langchain.namespaces.search import (
    ...     QdrantNamespaceSearchPipeline,
    ... )
    >>> pipeline = QdrantNamespaceSearchPipeline(
    ...     "config.yaml",
    ...     namespace="arc_train",
    ... )
    >>> results = pipeline.search("What is photosynthesis?", top_k=5)

See Also:
    - vectordb.langchain.namespaces.indexing.qdrant: Namespace-scoped indexing
    - vectordb.langchain.namespaces.qdrant: Core namespace implementation
"""

import logging
from typing import Any

from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class QdrantNamespaceSearchPipeline:
    """Qdrant search pipeline for namespace scenarios (LangChain).

    Performs semantic search within a single namespace using Qdrant's
    payload-based filtering for data isolation.

    Attributes:
        config: Validated configuration dictionary.
        namespace: Namespace to search within.
        embedder: LangChain embedder instance for query vectorization.
        pipeline: QdrantNamespacePipeline for namespace-specific operations.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> pipeline = QdrantNamespaceSearchPipeline(config, namespace="arc_train")
        >>> results = pipeline.search("machine learning", top_k=5)
    """

    def __init__(self, config_or_path: dict[str, Any] | str, namespace: str) -> None:
        """Initialize the namespace search pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
            namespace: Namespace to search within. Cannot be empty.

        Raises:
            ValueError: If namespace is empty or required config is missing.
        """
        if not namespace:
            raise ValueError("namespace cannot be empty")

        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")
        self.namespace = namespace

        self.embedder = EmbedderHelper.create_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.pipeline = QdrantNamespacePipeline(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
            collection_prefix=qdrant_config.get("collection_prefix", "ns_"),
        )

        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Qdrant namespace search pipeline for namespace: %s",
            namespace,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search within the namespace.

        Args:
            query: Search query text.
            top_k: Number of documents to return.
            filters: Optional metadata filters.

        Returns:
            Dictionary containing:
                - documents: List of Document objects
                - query: Original query string
                - namespace: The namespace that was searched
                - answer: Generated RAG answer if LLM configured (optional)
        """
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query for namespace %s: %s", self.namespace, query[:50])

        documents = self.pipeline.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            namespace=self.namespace,
        )
        logger.info(
            "Retrieved %d documents for namespace %s from Qdrant",
            len(documents),
            self.namespace,
        )

        result: dict[str, Any] = {
            "documents": documents,
            "query": query,
            "namespace": self.namespace,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer for namespace %s", self.namespace)

        return result
