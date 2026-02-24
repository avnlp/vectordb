"""Qdrant agentic RAG indexing pipeline (LangChain).

This module provides the indexing pipeline for agentic RAG using Qdrant as the
vector store backend. Qdrant's Rust-based implementation offers exceptional
performance and memory efficiency, making it ideal for resource-constrained
agentic RAG deployments and edge computing scenarios.

The indexing pipeline leverages Qdrant's payload-based architecture:
1. Document loading via DataloaderCatalog with metadata preservation
2. Embedding generation with batch optimization
3. Collection creation with payload indexing for filtered search

Qdrant-Specific Features for Agentic RAG:
    - Payload indices enable fast metadata filtering during agent queries
    - Quantization (scalar/binary) reduces memory footprint for large collections
    - Snapshot support enables backup/restore of agent knowledge bases
    - gRPC interface provides low-latency communication for iterative retrieval

Architecture Notes:
    Qdrant's efficient memory usage allows running multiple agentic RAG instances
    on modest hardware. The payload-based filtering integrates naturally with
    agent decision-making, allowing the agent to specify complex filter criteria
    based on its reasoning about query requirements.
"""

import logging
import uuid
from typing import Any

from qdrant_client.http.models import PointStruct

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class QdrantAgenticRAGIndexingPipeline:
    """Qdrant indexing pipeline for agentic RAG (LangChain).

    Loads documents, generates embeddings, creates Qdrant collection, and indexes
    documents for use in agentic RAG pipelines. Qdrant's performance characteristics
    make it well-suited for latency-sensitive agentic applications.

    The pipeline supports both Qdrant Cloud and self-hosted deployments through
    URL and API key configuration.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: QdrantVectorDB instance for vector storage operations.
        collection_name: Target Qdrant collection for document storage.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Validates configuration and initializes the Qdrant connection.
        Supports both Qdrant Cloud (https://<cluster>.cloud.qdrant.io) and
        self-hosted instances.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain 'qdrant' section with url, optional api_key,
                and collection_name settings.

        Raises:
            ValueError: If required Qdrant configuration is missing.

        Example:
            >>> pipeline = QdrantAgenticRAGIndexingPipeline("config.yaml")
            >>> result = pipeline.run()
            >>> print(f"Indexed {result['documents_indexed']} documents to Qdrant")
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Configure Qdrant connection
        qdrant_config = self.config["qdrant"]
        self.collection_name = qdrant_config.get("collection_name")

        self.db = QdrantVectorDB(
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key"),
            collection_name=self.collection_name,
        )

        logger.info("Initialized Qdrant agentic RAG indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs complete indexing workflow: loads documents, generates embeddings,
        creates or recreates the Qdrant collection with optimized configuration,
        and upserts all documents.

        Qdrant collections support payload indices that accelerate filtered
        searches. The agentic search pipeline can leverage these indices when
        the agent decides to apply metadata filters based on query analysis.

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

        # Create collection with payload indexing support
        # Payload indices enable efficient filtered search during agent queries
        recreate = self.config.get("qdrant", {}).get("recreate", False)
        dimension = len(embeddings[0]) if embeddings else 0
        self.db.create_collection(dimension=dimension, recreate=recreate)

        # Upsert documents with embeddings and metadata payloads
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "page_content": doc.page_content,
                    **doc.metadata,
                },
            )
            for doc, embedding in zip(docs, embeddings)
        ]

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.db.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
                wait=False,
            )
        logger.info("Indexed %d documents to Qdrant", len(docs))

        return {"documents_indexed": len(docs)}
