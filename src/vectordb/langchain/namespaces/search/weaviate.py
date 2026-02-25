"""Weaviate namespace search pipeline (LangChain).

This module provides a search pipeline for Weaviate vector database with
namespace support using Weaviate's tenant mechanism.

Example:
    >>> from vectordb.langchain.namespaces.search import (
    ...     WeaviateNamespaceSearchPipeline,
    ... )
    >>> pipeline = WeaviateNamespaceSearchPipeline(
    ...     "config.yaml",
    ...     namespace="arc_train",
    ... )
    >>> results = pipeline.search("What is photosynthesis?", top_k=5)

See Also:
    - vectordb.langchain.namespaces.indexing.weaviate: Namespace-scoped indexing
    - vectordb.langchain.namespaces.weaviate: Core namespace implementation
"""

import logging
from typing import Any

from vectordb.langchain.namespaces.weaviate import WeaviateNamespacePipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    HaystackToLangchainConverter,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class WeaviateNamespaceSearchPipeline:
    """Weaviate search pipeline for namespace scenarios (LangChain).

    Performs semantic search within a single namespace using Weaviate's
    tenant mechanism for data isolation.

    Attributes:
        config: Validated configuration dictionary.
        namespace: Namespace to search within.
        embedder: LangChain embedder instance for query vectorization.
        pipeline: WeaviateNamespacePipeline for namespace-specific operations.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> pipeline = WeaviateNamespaceSearchPipeline(config, namespace="arc_train")
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
        ConfigLoader.validate(self.config, "weaviate")
        self.namespace = namespace

        self.embedder = EmbedderHelper.create_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.pipeline = WeaviateNamespacePipeline(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
            collection_prefix=weaviate_config.get("collection_prefix", "ns_"),
        )

        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Weaviate namespace search pipeline for namespace: %s",
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
            vector=query_embedding,
            limit=top_k,
            filters=filters,
            return_documents=True,
        )
        documents = HaystackToLangchainConverter.convert(documents)
        logger.info(
            "Retrieved %d documents for namespace %s from Weaviate",
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
