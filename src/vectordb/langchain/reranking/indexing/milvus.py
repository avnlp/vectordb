"""Milvus reranking indexing pipeline (LangChain).

This module provides the indexing pipeline for Milvus vector database
with reranking support. Milvus is an open-source vector database designed
for AI applications with support for billion-scale vector collections.

The pipeline supports:
- Self-hosted Milvus instances (standalone or cluster)
- Collection-based organization with schema definition
- GPU acceleration for embedding computation
- Advanced indexing algorithms (IVF, HNSW, etc.)
"""

import logging
from typing import Any

from haystack.dataclasses import Document

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class MilvusRerankingIndexingPipeline:
    """Indexing pipeline for Milvus with reranking support.

    This pipeline loads documents, generates embeddings, and indexes them
    into Milvus for later reranked search.

    Milvus is ideal for:
    - Large-scale production deployments
    - Applications requiring GPU acceleration
    - Complex partitioning and sharding strategies
    - Enterprise-grade vector search with high availability

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model instance
        db: MilvusVectorDB instance for database operations
        collection_name: Name of the Milvus collection

    Example:
        >>> pipeline = MilvusRerankingIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents to Milvus")

    Configuration Requirements:
        The config file must specify:
        - milvus.host: Milvus server host (default: localhost)
        - milvus.port: Milvus server port (default: 19530)
        - milvus.collection_name: Target collection name (default: "reranking")
        - embeddings: Embedding model configuration
        - dataloader: Data source configuration
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Milvus indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the Milvus vector database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
            ConnectionError: If unable to connect to Milvus.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get("collection_name", "reranking")

        logger.info("Initialized Milvus reranking indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and inserts all documents into the Milvus collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.

        Pipeline Steps:
            1. Load documents from configured data source
            2. Generate embeddings for all documents using embedder
            3. Attach embeddings to Haystack Document objects
            4. Insert documents to Milvus collection
            5. Return count of indexed documents
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

        haystack_docs = [
            Document(
                content=doc.page_content,
                embedding=embedding,
                meta=doc.metadata or {},
            )
            for doc, embedding in zip(docs, embeddings)
        ]

        self.db.insert_documents(
            documents=haystack_docs,
            collection_name=self.collection_name,
        )
        num_indexed = len(haystack_docs)
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {"documents_indexed": num_indexed}
