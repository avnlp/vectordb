"""Pinecone reranking indexing pipeline.

This module provides an indexing pipeline for preparing document collections
in Pinecone for two-stage retrieval with reranking.

Indexing for Reranking:
    Documents are embedded using bi-encoder models and stored in Pinecone.
    These embeddings enable fast approximate nearest neighbor retrieval during
    the first stage of reranking search. The cross-encoder used in the second
    stage operates on text, not embeddings, so only the bi-encoder output
    needs to be indexed.

Pipeline Steps:
    1. Load documents from configured data sources (limit optional)
    2. Generate dense embeddings using bi-encoder models
    3. Create or recreate Pinecone index with appropriate dimensions
    4. Upsert embedded documents with metadata for filtering

Pinecone-Specific Considerations:
    - Serverless indexes auto-scale but may have cold start latency
    - Pod-based indexes offer predictable performance
    - Metadata supports filtering but has size limits
    - Namespaces enable logical separation within an index

Configuration Requirements:
    - pinecone.api_key: Authentication token
    - pinecone.index_name: Target index name
    - pinecone.namespace: Optional logical partition
    - pinecone.metric: Similarity metric (cosine, euclidean, dotproduct)
    - embedder: Model configuration for bi-encoder
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class PineconeRerankingIndexingPipeline:
    """Pinecone indexing pipeline for reranking document collections.

    Prepares document collections for two-stage retrieval by generating
    bi-encoder embeddings and storing them in Pinecone. The cross-encoder
    reranker used during search operates on raw text, so only the initial
    retrieval stage requires indexed embeddings.

    Attributes:
        config: Pipeline configuration dict.
        embedder: Bi-encoder component for document embedding generation.
        dimension: Embedding dimension inferred from embedder configuration.
        db: PineconeVectorDB instance for index management.
        index_name: Name of the Pinecone index to create/use.
        namespace: Optional namespace for logical data separation.

    Example:
        >>> pipeline = PineconeRerankingIndexingPipeline("pinecone_config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Successfully indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - pinecone: API key, index name, namespace, metric
                - embedder: Provider, model, dimensions for bi-encoder
                - dataloader: Dataset configuration and optional limit

        Raises:
            ValueError: If required config sections are missing.
            KeyError: If pinecone.api_key is not provided in config.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)
        self.dimension = EmbedderFactory.get_embedding_dimension(self.embedder)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        logger.info("Initialized Pinecone reranking indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Loads documents from configured data sources, generates bi-encoder
        embeddings, creates the Pinecone index if needed, and upserts all
        documents with their embeddings and metadata.

        Returns:
            Dict with 'documents_indexed' count indicating success.

        Raises:
            RuntimeError: If embedding generation or Pinecone upsert fails.
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_haystack()
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        embedded_docs = self.embedder.run(documents=documents)["documents"]
        logger.info("Generated embeddings for %d documents", len(embedded_docs))

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        num_indexed = self.db.upsert(
            documents=embedded_docs,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
