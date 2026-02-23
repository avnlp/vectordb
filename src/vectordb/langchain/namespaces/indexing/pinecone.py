"""Pinecone namespace indexing pipeline (LangChain).

This module provides an indexing pipeline for Pinecone vector database with
namespace support. Namespaces ensure complete data isolation between
different data partitions using Pinecone's namespace mechanism.

Namespace Strategy:
    Pinecone namespaces provide isolation:
    - Each namespace's data is stored in a separate Pinecone namespace
    - Queries are scoped to a single namespace, preventing cross-namespace access
    - Namespace deletion removes all data within that namespace

Pipeline Flow:
    1. Validate namespace (cannot be empty)
    2. Load configuration and validate Pinecone settings
    3. Initialize embedder for dense vector generation
    4. Initialize PineconeNamespacePipeline for namespace operations
    5. Load documents from configured data source
    6. Generate embeddings for all documents
    7. Create Pinecone index (shared across all namespaces)
    8. Index documents to specific namespace

Example:
    >>> from vectordb.langchain.namespaces.indexing import (
    ...     PineconeNamespaceIndexingPipeline,
    ... )
    >>> pipeline = PineconeNamespaceIndexingPipeline(
    ...     "config.yaml",
    ...     namespace="arc_train",
    ... )
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} docs for {result['namespace']}")

See Also:
    - vectordb.langchain.namespaces.search.pinecone: Namespace-scoped search
    - vectordb.langchain.namespaces.pinecone: Core namespace implementation
"""

import logging
from typing import Any

from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.namespaces.pinecone import PineconeNamespacePipeline
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class PineconeNamespaceIndexingPipeline:
    """Pinecone indexing pipeline for namespace scenarios (LangChain).

    Indexes documents into a specific namespace, ensuring data isolation.
    Uses Pinecone's namespace mechanism for efficient isolation.

    Attributes:
        config: Validated configuration dictionary.
        namespace: Target namespace for indexing.
        embedder: LangChain embedder instance for generating document vectors.
        pipeline: PineconeNamespacePipeline for namespace-specific operations.
        index_name: Name of the Pinecone index.
        dimension: Vector dimension matching the embedder output.

    Example:
        >>> pipeline = PineconeNamespaceIndexingPipeline(config, namespace="arc_train")
        >>> result = pipeline.run()
    """

    def __init__(self, config_or_path: dict[str, Any] | str, namespace: str) -> None:
        """Initialize the namespace indexing pipeline.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
            namespace: Target namespace for indexing. Cannot be empty.

        Raises:
            ValueError: If namespace is empty or required config is missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
        """
        if not namespace:
            raise ValueError("namespace cannot be empty")

        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")
        self.namespace = namespace

        self.embedder = EmbedderHelper.create_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.pipeline = PineconeNamespacePipeline(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
            dimension=pinecone_config.get("dimension", 384),
        )

        self.index_name = pinecone_config.get("index_name")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info(
            "Initialized Pinecone namespace indexing pipeline for namespace: %s",
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

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.pipeline.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        result = self.pipeline.index_documents(
            documents=docs,
            embeddings=embeddings,
            namespace=self.namespace,
        )

        num_indexed = result.data.get("count", 0) if result.data else 0
        logger.info(
            "Indexed %d documents for namespace %s to Pinecone",
            num_indexed,
            self.namespace,
        )

        return {"documents_indexed": num_indexed, "namespace": self.namespace}
