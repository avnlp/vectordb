"""Weaviate semantic search indexing pipeline.

This pipeline provides document indexing for Weaviate vector database,
enabling semantic similarity search through dense vector embeddings with
schema-based organization and GraphQL interface.

Weaviate-Specific Considerations:
    - Weaviate combines vector search with semantic (GraphQL) capabilities
    - Schema-based with automatic vectorization support
    - Supports modular AI integrations (vectorization, generative, qna)
    - Can be self-hosted or used as a managed service (WCS)
    - Multi-tenancy with tenant isolation

Indexing Pipeline Steps:
    1. Load documents: Fetch from dataset via DataloaderCatalog
    2. Generate embeddings: Use configured embedder to create dense vectors
    3. Create class: Initialize Weaviate class/collection with schema
    4. Insert documents: Store vectors and metadata in Weaviate

Configuration (YAML):
    Required sections:
        - weaviate.url: Weaviate server URL (e.g., "http://localhost:8080")
        - weaviate.api_key: Authentication credentials
        - weaviate.class_name: Name of the class/collection to create
        - embeddings.model: HuggingFace model path for embeddings
        - dataloader.type: Dataset type (e.g., "triviaqa")

    Optional settings:
        - weaviate.recreate: Whether to drop and recreate existing class
        - dataloader.limit: Optional limit on documents to process

    Example config:
        weaviate:
          url: "https://my-cluster.weaviate.cloud"
          api_key: "${WEAVIATE_API_KEY}"
          class_name: "SemanticSearch"
          recreate: false
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        dataloader:
          type: "triviaqa"
          limit: 1000

Usage:
    >>> from vectordb.haystack.semantic_search import WeaviateSemanticIndexingPipeline
    >>> pipeline = WeaviateSemanticIndexingPipeline("config.yaml")
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

    Both implementations use the same underlying WeaviateVectorDB class for
database operations, ensuring consistent behavior across frameworks.

Note:
    Weaviate uses a schema-based approach. The class is created
with properties for content and metadata, plus a vector index.
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory


logger = logging.getLogger(__name__)


class WeaviateSemanticIndexingPipeline:
    """Weaviate indexing pipeline for semantic search.

        Loads documents, generates embeddings, creates class, and indexes.

        This pipeline follows the standard 3-stage indexing pattern:
        1. Load documents from the configured dataset
        2. Generate embeddings using the configured embedder
        3. Create class and insert documents to Weaviate

    Attributes:
            config: Validated configuration dictionary.
            embedder: Haystack document embedder component.
            db: WeaviateVectorDB instance for database operations.
            class_name: Name of the Weaviate class/collection.

    Note:
            Weaviate uses a schema-based approach. The class is created
    with properties for content and metadata, plus a vector index.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderFactory.create_document_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config.get("url", "http://localhost:8080"),
            api_key=weaviate_config.get("api_key", ""),
        )

        self.class_name = weaviate_config["class_name"]

        logger.info("Initialized Weaviate indexing pipeline")

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

        recreate = self.config.get("weaviate", {}).get("recreate", False)
        self.db.create_class(
            class_name=self.class_name,
            recreate=recreate,
        )

        # Insert documents
        self.db.insert_documents(
            documents=embedded_docs,
            class_name=self.class_name,
        )
        logger.info("Indexed %d documents to Weaviate", len(embedded_docs))

        return {"documents_indexed": len(embedded_docs)}
