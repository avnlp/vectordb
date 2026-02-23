"""Milvus agentic RAG indexing pipeline (LangChain).

This module provides the indexing pipeline for agentic RAG using Milvus as the
vector store backend. Milvus offers enterprise-grade scalability and advanced
search capabilities that support complex agentic RAG workflows requiring
high-throughput retrieval and sophisticated filtering.

The indexing pipeline leverages Milvus's distributed architecture:
1. Document loading via DataloaderCatalog with batch processing
2. Embedding generation optimized for Milvus's vector ingestion
3. Collection creation with partition support for large-scale datasets

Milvus-Specific Advantages for Agentic RAG:
    - Partitioning enables time-based or category-based document organization,
      allowing the agent to target specific document subsets during search
    - Hybrid search (dense + sparse vectors) improves retrieval for queries
      with specific keywords that semantic search might miss
    - GPU index acceleration reduces latency for real-time agent interactions
    - Multi-replica support ensures availability during high query volumes

Architecture Notes:
    Milvus's separation of storage and compute aligns well with agentic RAG's
    iterative retrieval pattern. The search pipeline can scale query nodes
    independently from index nodes, maintaining performance during agent
    reasoning cycles that may involve multiple retrieval rounds.
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class MilvusAgenticRAGIndexingPipeline:
    """Milvus indexing pipeline for agentic RAG (LangChain).

    Loads documents, generates embeddings, creates Milvus collection, and indexes
    documents for use in agentic RAG pipelines. Milvus's scalability makes it
    suitable for large-scale agentic RAG deployments with millions of documents.

    The pipeline supports both Zilliz Cloud (managed Milvus) and self-hosted
    Milvus deployments through URI and token configuration.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: MilvusVectorDB instance for vector storage operations.
        collection_name: Target Milvus collection for document storage.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the Milvus connection.
        Supports both Zilliz Cloud (managed) and self-hosted Milvus through
        the uri parameter. Token authentication is required for Zilliz Cloud.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'milvus' section with uri, optional token,
                and collection_name settings.

        Raises:
            ValueError: If required Milvus configuration is missing.

        Example:
            >>> pipeline = MilvusAgenticRAGIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents to Milvus")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Configure Milvus connection
        # URI format: "https://<cluster>.zillizcloud.com" for Zilliz Cloud
        # or "http://localhost:19530" for self-hosted
        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri"),
            token=milvus_config.get("token"),
        )

        self.collection_name = milvus_config.get("collection_name")

        logger.info("Initialized Milvus agentic RAG indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs complete indexing workflow: loads documents, generates embeddings,
        creates or recreates the Milvus collection with optimized schema, and
        upserts all documents.

        Milvus collections support dynamic fields and complex schemas that can
        enhance agentic RAG through rich metadata filtering. The recreate option
        allows schema evolution during development.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed

        Raises:
            Exception: If collection creation or document upsert fails.
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

        # Create collection with dynamic schema support
        # Dynamic fields enable flexible metadata without schema migrations
        recreate = self.config.get("milvus", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=recreate,
        )

        # Upsert documents to collection
        # Milvus handles large-scale batch ingestion efficiently
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
