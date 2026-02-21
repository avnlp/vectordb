"""Milvus hybrid search pipeline (LangChain).

Implements hybrid search using Milvus's native sparse vector support. Milvus
2.3+ supports separate sparse and dense vector fields within the same
collection, enabling efficient hybrid similarity search.

The hybrid search mechanism:
    1. Query is embedded using both dense and sparse embedders
    2. Dense embedding captures semantic meaning (floating-point vector)
    3. Sparse embedding captures exact keyword matches (sparse vector format)
    4. Milvus performs ANN search on both vector types
    5. Results are ranked by weighted combination of similarities

Sparse vector format:
    Milvus accepts sparse vectors as {index: value} dictionaries where:
    - index: Integer dimension (vocabulary position)
    - value: Float weight (typically TF-IDF score)
    Only non-zero entries are stored for efficiency.

Alpha parameter (0.0 to 1.0):
    - 1.0: Pure dense/semantic search
    - 0.0: Pure sparse/keyword search
    - 0.5: Balanced hybrid (default)

Milvus advantages:
    - High-performance sparse vector operations
    - GPU acceleration support for dense vectors
    - Distributed deployment options
    - Strong consistency guarantees
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class MilvusHybridSearchPipeline:
    """Milvus hybrid (dense + sparse) search pipeline.

    Combines dense semantic embeddings with sparse lexical embeddings using
    Milvus's native hybrid search capabilities. Leverages Milvus's optimized
    sparse vector storage and retrieval.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for semantic vector generation.
        sparse_embedder: Embedder for sparse TF-IDF vector generation.
        db: MilvusVectorDB instance for vector operations.
        collection_name: Target Milvus collection name.
        alpha: Fusion weight between dense and sparse (0.0-1.0).
        llm: Optional language model for RAG answer generation.

    Example:
        >>> pipeline = MilvusHybridSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="attention mechanism paper",
        ...     top_k=10,
        ...     filters={"year": {"$gte": 2020}},
        ... )
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain milvus section with connection details and
                optional collection_name, alpha settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Configuration Schema:
            milvus:
              host: "localhost"
              port: 19530
              collection_name: "hybrid-collection"
              alpha: 0.5
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
            collection_name=milvus_config.get("collection_name"),
        )

        self.collection_name = milvus_config.get("collection_name")
        self.alpha = milvus_config.get("alpha", 0.5)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Milvus hybrid search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search combining dense and sparse embeddings.

        Generates both dense (semantic) and sparse (lexical) embeddings for
        the query, then executes Milvus's native hybrid search. Optionally
        generates a RAG answer if an LLM was configured.

        Args:
            query: Search query text to embed and search.
            top_k: Maximum number of results to return. Defaults to 10.
            filters: Optional metadata filters as dictionary for pre-filtering.
                Supports Milvus's expression-based filters.

        Returns:
            Dictionary containing:
                - documents: List of retrieved Document objects
                - query: Original query string
                - answer: Generated RAG answer (only if LLM configured)

        Raises:
            RuntimeError: If database connection fails during search.

        Sparse Embedding Details:
            Sparse embeddings use a {dimension_index: weight} format optimized
            for Milvus's sparse vector field type. The SparseEmbedder handles
            TF-IDF transformation and format conversion automatically.
        """
        dense_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        sparse_embedding = self.sparse_embedder.embed_query(query)
        logger.info("Generated hybrid embeddings for query: %s...", query[:50])

        documents = self.db.hybrid_search(
            query_embedding=dense_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
            alpha=self.alpha,
        )
        logger.info("Retrieved %d documents from Milvus", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
