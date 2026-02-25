"""Chroma query enhancement indexing pipeline (LangChain).

This module provides the indexing pipeline for query enhancement using Chroma
as the vector database backend. The indexing phase prepares documents for
enhanced retrieval by storing them with dense vector embeddings.

Indexing Pipeline Architecture:
    The pipeline follows a standard three-phase pattern:
        1. Document Loading: Load documents from configured data source
        2. Embedding Generation: Generate dense vectors for all documents
        3. Vector Store Indexing: Store documents with embeddings in Chroma

    Query enhancement happens at search time, so indexing is identical to
    standard semantic search indexing. The difference is in how queries
    are processed during retrieval.

Query Enhancement Prerequisites:
    - Documents are embedded using the same model as during search
    - Collection name must match between indexing and search pipelines
    - Embedder configuration must be consistent across both phases

    Mismatched embedders or collection names will produce poor results
    because query embeddings won't match document embeddings.

Configuration:
    Requires standard Chroma and embedder configuration:
        chroma:
          persist_dir: "./chroma_db"
          collection_name: "documents"
          recreate: false  # Whether to recreate collection

        embedder:
          model: "sentence-transformers/all-MiniLM-L6-v2"
          device: "cpu"

        dataloader:
          type: "triviaqa"
          limit: 100  # Optional document limit

Example:
    >>> pipeline = ChromaQueryEnhancementIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
    >>> # Search pipeline can now use query enhancement
    >>> from vectordb.langchain.query_enhancement.search.chroma import (
    ...     ChromaQueryEnhancementSearchPipeline,
    ... )
    >>> searcher = ChromaQueryEnhancementSearchPipeline("config.yaml")

See Also:
    vectordb.langchain.query_enhancement.search.chroma: Search pipeline
    vectordb.langchain.components.query_enhancer: Core enhancement logic
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class ChromaQueryEnhancementIndexingPipeline:
    """Chroma indexing pipeline for query-enhanced retrieval (LangChain).

    Prepares document collections for query-enhanced search by indexing
    documents with their dense vector embeddings. The resulting collection
    can be queried with multi-query, or step-query, HyDE enhancement strategies.

    This pipeline is functionally identical to standard semantic search indexing.
    Query enhancement is applied at search time using the same embedded documents.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Configured embedding model for document vectorization.
        db: ChromaVectorDB instance for local vector storage.
        collection_name: Name of Chroma collection for document storage.

    Example:
        >>> config = {
        ...     "chroma": {
        ...         "persist_dir": "./chroma_db",
        ...         "collection_name": "documents",
        ...     },
        ...     "embedder": {"model": "all-MiniLM-L6-v2"},
        ...     "dataloader": {"type": "triviaqa", "limit": 100},
        ... }
        >>> pipeline = ChromaQueryEnhancementIndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize query enhancement indexing pipeline from configuration.

        Validates configuration and initializes Chroma connection and
        embedding model for document processing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain chroma and embedder sections.

        Raises:
            ValueError: If required configuration is missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_dir=chroma_config.get("persist_dir"),
        )

        self.collection_name = chroma_config.get("collection_name")

        logger.info(
            "Initialized Chroma query enhancement indexing pipeline (LangChain)"
        )

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Performs the complete indexing workflow: loads documents from the
        configured data source, generates embeddings, creates the Chroma
        collection, and upserts all documents with their vectors.

        Returns:
            Dictionary containing:
                - documents_indexed: Number of documents successfully indexed

        Example:
            >>> result = pipeline.run()
            >>> print(f"Successfully indexed {result['documents_indexed']} documents")
        """
        # Load documents with optional limit
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

        # Generate embeddings for all documents
        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d documents", len(docs))

        # Create or recreate collection based on configuration
        recreate = self.config.get("chroma", {}).get("recreate", False)
        self.db.create_collection(
            collection_name=self.collection_name,
            recreate=recreate,
        )

        # Upsert documents with embeddings to Chroma
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
