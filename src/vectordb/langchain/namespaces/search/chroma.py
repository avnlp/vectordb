"""Chroma namespace search pipeline (LangChain).

This module provides a search pipeline for Chroma vector database with
namespace support using separate collections.

Example:
    >>> from vectordb.langchain.namespaces.search import (
    ...     ChromaNamespaceSearchPipeline,
    ... )
    >>> pipeline = ChromaNamespaceSearchPipeline(
    ...     "config.yaml",
    ...     namespace="arc_train",
    ... )
    >>> results = pipeline.search("What is photosynthesis?", top_k=5)

See Also:
    - vectordb.langchain.namespaces.indexing.chroma: Namespace-scoped indexing
    - vectordb.langchain.namespaces.chroma: Core namespace implementation
"""

import logging
from typing import Any

from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
)
from vectordb.utils.chroma_document_converter import ChromaDocumentConverter


logger = logging.getLogger(__name__)


class ChromaNamespaceSearchPipeline:
    """Chroma search pipeline for namespace scenarios (LangChain).

    Performs semantic search within a single namespace using Chroma's
    collection mechanism for data isolation.

    Attributes:
        config: Validated configuration dictionary.
        namespace: Namespace to search within.
        embedder: LangChain embedder instance for query vectorization.
        pipeline: ChromaNamespacePipeline for namespace-specific operations.
        llm: Optional LangChain LLM for RAG answer generation.

    Example:
        >>> pipeline = ChromaNamespaceSearchPipeline(config, namespace="arc_train")
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
        ConfigLoader.validate(self.config, "chroma")
        self.namespace = namespace

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.pipeline = ChromaNamespacePipeline(
            path=chroma_config.get("path", "./chroma_data"),
            collection_prefix=chroma_config.get("collection_prefix", "ns_"),
        )

        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Chroma namespace search pipeline for namespace: %s",
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

        collection_name = self.pipeline._get_collection_name(self.namespace)
        self.pipeline.db._get_collection(collection_name)
        results_dict = self.pipeline.db.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=filters,
        )
        documents = (
            ChromaDocumentConverter.convert_query_results_to_langchain_documents(
                results_dict
            )
        )
        logger.info(
            "Retrieved %d documents for namespace %s from Chroma",
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
