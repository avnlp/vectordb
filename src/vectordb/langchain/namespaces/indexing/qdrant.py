"""Qdrant namespace indexing pipeline (LangChain).

This module provides an indexing pipeline for Qdrant vector database with
namespace support using payload-based filtering.

Example:
    >>> from vectordb.langchain.namespaces.indexing import (
    ...     QdrantNamespaceIndexingPipeline,
    ... )
    >>> pipeline = QdrantNamespaceIndexingPipeline(
    ...     "config.yaml",
    ...     namespace="arc_train",
    ... )
    >>> result = pipeline.run()

See Also:
    - vectordb.langchain.namespaces.search.qdrant: Namespace-scoped search
    - vectordb.langchain.namespaces.qdrant: Core namespace implementation
"""

import logging
from typing import Any

from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class QdrantNamespaceIndexingPipeline:
    """Qdrant indexing pipeline for namespace scenarios (LangChain).

    Indexes documents into a specific namespace using Qdrant's
    payload-based filtering for data isolation.

    Attributes:
        config: Validated configuration dictionary.
        namespace: Target namespace for indexing.
        embedder: LangChain embedder instance for generating document vectors.
        pipeline: QdrantNamespacePipeline for namespace-specific operations.

    Example:
        >>> pipeline = QdrantNamespaceIndexingPipeline(config, namespace="arc_train")
        >>> result = pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str, namespace: str) -> None:
        """Initialize the namespace indexing pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
            namespace: Target namespace for indexing. Cannot be empty.

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

        logger.info(
            "Initialized Qdrant namespace indexing pipeline for namespace: %s",
            namespace,
        )

    def run(self) -> dict[str, Any]:
        """Execute the namespace indexing pipeline.

        Returns:
            Dictionary with:
                - documents_indexed: Number of documents indexed (int)
                - namespace: The namespace that was indexed (str)
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_langchain()
        logger.info(
            "Loaded %d documents for namespace %s", len(documents), self.namespace
        )

        if not documents:
            logger.warning("No documents to index for namespace: %s", self.namespace)
            return {"documents_indexed": 0, "namespace": self.namespace}

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info(
            "Generated embeddings for %d documents for namespace %s",
            len(docs),
            self.namespace,
        )

        result = self.pipeline.index_documents(
            documents=docs,
            embeddings=embeddings,
            namespace=self.namespace,
        )

        num_indexed = result.data.get("count", 0) if result.data else 0
        logger.info(
            "Indexed %d documents for namespace %s to Qdrant",
            num_indexed,
            self.namespace,
        )

        return {"documents_indexed": num_indexed, "namespace": self.namespace}
