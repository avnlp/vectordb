"""Pinecone semantic search indexing pipeline.

This pipeline provides document indexing for Pinecone's managed vector database,
enabling semantic similarity search through dense vector embeddings.

Pinecone-Specific Considerations:
    - Pinecone is a managed cloud service requiring API key authentication
    - Indexes are created with a specified dimension and metric
      (cosine, dotproduct, euclidean)
    - Supports namespaces for logical partitioning within an index
    - Serverless and pod-based deployment options available
    - Real-time index updates without reindexing

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create index: Initialize Pinecone index with proper dimension/metric
    4. Upsert documents: Store vectors and metadata in Pinecone

Configuration (YAML):
    Required sections:
        - pinecone.api_key: Pinecone API key
        - pinecone.index_name: Name of the index to create
        - embeddings.model: HuggingFace model path for embeddings
        - dataloader.type: Dataset type (e.g., "triviaqa")

    Optional settings:
        - pinecone.namespace: Namespace for document organization
        - pinecone.metric: Similarity metric (cosine, dotproduct, euclidean)
        - pinecone.recreate: Whether to drop and recreate existing index
        - dataloader.limit: Optional limit on documents to process

    Example config:
        pinecone:
          api_key: "${PINECONE_API_KEY}"
          index_name: "semantic-search"
          namespace: "production"
          metric: "cosine"
          recreate: false
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        dataloader:
          type: "triviaqa"
          limit: 1000

Usage:
    >>> from vectordb.haystack.semantic_search import PineconeSemanticIndexingPipeline
    >>> pipeline = PineconeSemanticIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")

Comparison with LangChain:
    Haystack Integration (this module):
        - Uses native Haystack Document format and embedders
        - Pipeline-based architecture with clear data flow
        - Built-in dataset loading through DataloaderCatalog

    LangChain Integration (vectordb.langchain):
        - Uses LangChain Document format
        - Chain-based composition
        - More flexible but requires more configuration

    Both implementations use the same underlying PineconeVectorDB class for
database operations, ensuring consistent behavior across frameworks.

Note:
    Pinecone indexes cannot be modified after creation (dimension/metric
    are immutable). The recreate flag drops and recreates the index.
"""

import logging
from typing import Any

from vectordb import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class PineconeSemanticIndexingPipeline:
    """Pinecone indexing pipeline for semantic search.

    Loads documents, generates embeddings, creates index, and indexes.

    This pipeline follows the standard 3-stage indexing pattern:
    1. Load documents from the configured dataset
    2. Generate embeddings using the configured embedder
    3. Create index and upsert documents to Pinecone

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack document embedder component.
        db: PineconeVectorDB instance for database operations.
        index_name: Name of the Pinecone index.
        namespace: Optional namespace for document organization.
        dimension: Vector dimension (must match embedding model).

    Note:
        Pinecone indexes cannot be modified after creation (dimension/metric
        are immutable). The recreate flag drops and recreates the index.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone indexing pipeline")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Returns:
            Dict with 'documents_indexed' count.
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

        # Upsert documents
        num_indexed = self.db.upsert(
            documents=embedded_docs,
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone", num_indexed)

        return {"documents_indexed": num_indexed}
