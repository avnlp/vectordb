"""Pinecone agentic RAG indexing pipeline (LangChain).

This module provides the indexing pipeline for agentic RAG using Pinecone as the
managed vector store backend. Pinecone's cloud-native architecture provides the
low-latency search and high availability required for production agentic RAG
deployments where response time directly impacts user experience.

The indexing pipeline follows the standard three-phase pattern with Pinecone-
specific optimizations:
1. Document loading via DataloaderCatalog
2. Embedding generation with dimension alignment to Pinecone index
3. Vector store indexing with namespace support for multi-tenant scenarios

Pinecone-Specific Considerations:
    - Index dimension must match embedding model output (default 384 for
      sentence-transformers/all-MiniLM-L6-v2)
    - Namespaces enable logical separation of documents for different agents
    - Serverless indexes auto-scale, but provisioning requires planning
    - Metadata filtering during agentic search requires indexed metadata fields

Architecture Notes:
    Pinecone's managed service eliminates operational overhead but introduces
    network latency. The agentic search pipeline mitigates this through
    aggressive document compression after retrieval, reducing the number of
    round-trips needed during multi-turn agent interactions.
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class PineconeAgenticRAGIndexingPipeline:
    """Pinecone indexing pipeline for agentic RAG (LangChain).

    Loads documents, generates embeddings, creates Pinecone index, and indexes
    documents for use in agentic RAG pipelines. Pinecone's managed infrastructure
    provides the reliability and performance characteristics needed for production
    agentic RAG systems.

    The pipeline handles index creation with configurable dimension and metric
    settings. The dimension must align with the chosen embedding model to prevent
    runtime errors during upsert operations.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: PineconeVectorDB instance for cloud vector storage.
        index_name: Target Pinecone index name.
        namespace: Logical namespace for document isolation.
        dimension: Vector dimension matching the embedding model.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the Pinecone connection.
        The dimension parameter is critical and must match the embedding model's
        output size. Default is 384 for MiniLM models, 768 for larger models.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'pinecone' section with api_key, index_name,
                and optional namespace and dimension settings.

        Raises:
            ValueError: If required Pinecone configuration is missing.

        Example:
            >>> pipeline = PineconeAgenticRAGIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents to Pinecone")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Configure Pinecone connection with API authentication
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        # Dimension must match embedding model output (384 for MiniLM, 768 for larger)
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone agentic RAG indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs complete indexing workflow: loads documents, generates embeddings,
        creates or recreates the Pinecone index with proper dimension configuration,
        and upserts all documents to the specified namespace.

        The recreate option is useful for development but should be used cautiously
        in production as it destroys existing data. Namespaces provide a safer
        mechanism for incremental updates without affecting other document sets.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed

        Raises:
            Exception: If index creation fails or dimension mismatch occurs.
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
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d documents", len(docs))

        # Create index with specified dimension and similarity metric
        # Cosine similarity is the default and works well for semantic search
        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        # Upsert to namespace for logical document separation
        # Namespaces enable multi-tenant agentic RAG without separate indexes
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
