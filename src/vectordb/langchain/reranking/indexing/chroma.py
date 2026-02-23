"""Chroma reranking indexing pipeline (LangChain).

This module provides the indexing pipeline for Chroma vector database with
reranking support. Reranking is a two-stage retrieval strategy where:

1. First Stage (Retrieval): Initial broad search using embeddings to retrieve
   a larger set of candidate documents (typically top 50-100)
2. Second Stage (Reranking): Cross-encoder scoring to reorder candidates
   based on finer-grained relevance to the query

Why Reranking:
    - Embeddings optimize for overall semantic similarity, sometimes missing
      nuanced relevance that cross-encoders can detect
    - Cross-encoders process query-document pairs directly, enabling deeper
      relevance assessment
    - Trade-off: More computationally expensive but higher precision results

Reranking Pipeline Architecture:
    1. Indexing: Store documents with embeddings (this module)
    2. Initial Retrieval: Broad embedding-based search (50-100 candidates)
    3. Reranking: Score candidates with cross-encoder, return top-k

Chroma as First-Stage Retriever:
    Chroma is ideal for first-stage retrieval because:
    - Local persistent storage with configurable path
    - Collection-based document organization
    - Efficient local embedding computation
    - Fast prototyping without cloud dependencies

    For reranking, Chroma retrieves a broad candidate set which is then
    refined by a cross-encoder model (e.g., BAAI/bge-reranker-base).

Configuration:
    chroma:
      path: "./chroma_data"  # Local directory for persistence
      collection_name: "reranking"  # Collection for reranking pipeline

    embedder:
      model: "sentence-transformers/all-MiniLM-L6-v2"  # First-stage retrieval

    cross_encoder:
      model: "BAAI/bge-reranker-base"  # Second-stage reranking

    dataloader:
      type: "triviaqa"
      limit: 100

Example:
    >>> from vectordb.langchain.reranking.indexing.chroma import (
    ...     ChromaRerankingIndexingPipeline,
    ... )
    >>> pipeline = ChromaRerankingIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

    Search pipeline uses the same collection for initial retrieval:
    >>> from vectordb.langchain.reranking.search.chroma import (
    ...     ChromaRerankingSearchPipeline,
    ... )
    >>> searcher = ChromaRerankingSearchPipeline("config.yaml")
    >>> results = searcher.search("machine learning", top_k=10)

See Also:
    vectordb.langchain.reranking.search.chroma: Reranking search pipeline
    vectordb.langchain.components.cross_encoder: Cross-encoder reranking
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


class ChromaReankingIndexingPipeline:
    """Indexing pipeline for Chroma with reranking support.

    This pipeline loads documents, generates embeddings, and indexes them
    into a local Chroma collection for later reranked search.

    Chroma is ideal for:
    - Local development and testing
    - Prototyping reranking pipelines
    - Small to medium document collections
    - Offline or air-gapped environments

    Attributes:
        config: Loaded configuration dictionary
        embedder: Initialized embedding model instance
        db: ChromaVectorDB instance for database operations
        collection_name: Name of the Chroma collection

    Example:
        >>> pipeline = ChromaReankingIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents to Chroma")

    Configuration Requirements:
        The config file must specify:
        - chroma.path: Local directory for Chroma persistence
        - chroma.collection_name: Target collection name (default: "reranking")
        - embedder: Embedding model configuration
        - dataloader: Data source configuration
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the Chroma indexing pipeline.

        Loads configuration, initializes the embedding model, and connects
        to the local Chroma database.

        Args:
            config_or_path: Either a configuration dictionary or path to
                a YAML configuration file.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path", "./chroma_data"),
        )

        self.collection_name = chroma_config.get("collection_name", "reranking")

        logger.info("Initialized Chroma reranking indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute the complete indexing pipeline.

        Loads documents from the configured data source, generates embeddings,
        and upserts all documents into the Chroma collection.

        Returns:
            Dictionary with indexing results:
            - documents_indexed: Number of documents successfully indexed (int)

        Raises:
            RuntimeError: If embedding generation or indexing fails.

        Pipeline Steps:
            1. Load documents from configured data source
            2. Generate embeddings for all documents using embedder
            3. Upsert documents with embeddings to Chroma collection
            4. Return count of indexed documents
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

        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
