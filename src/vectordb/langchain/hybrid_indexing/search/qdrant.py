"""Qdrant hybrid search pipeline (LangChain).

Implements hybrid search using Qdrant's native sparse vector support. Qdrant
allows storing and querying both dense and sparse vectors in the same
payload, enabling true hybrid similarity search.

The hybrid search mechanism:
    1. Query is embedded using both dense and sparse embedders
    2. Dense embedding captures semantic meaning (neural network-based)
    3. Sparse embedding captures exact keyword matches (TF-IDF style)
    4. Qdrant computes similarity for both vector types
    5. Results are ranked by weighted combination of similarities

Sparse vector format:
    Qdrant expects sparse vectors as dictionaries mapping dimension indices
    to float values (non-zero entries only). The SparseEmbedder generates
    this format from text using scikit-learn's TF-IDF vectorization.

Alpha parameter (0.0 to 1.0):
    - 1.0: Pure dense/semantic search
    - 0.0: Pure sparse/keyword search
    - 0.5: Balanced hybrid (default)

Qdrant advantages:
    - Native sparse vector storage (efficient compression)
    - Query-time fusion with configurable weights
    - Filter support on metadata during hybrid search
    - Open-source with self-hosting options
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class QdrantHybridSearchPipeline:
    """Qdrant hybrid (dense + sparse) search pipeline.

    Combines dense semantic embeddings with sparse lexical embeddings using
    Qdrant's native hybrid search capabilities. Supports configurable alpha
    weighting and metadata filtering.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for semantic vector generation.
        sparse_embedder: Embedder for sparse TF-IDF vector generation.
        db: QdrantVectorDB instance for vector operations.
        collection_name: Target Qdrant collection name.
        alpha: Fusion weight between dense and sparse (0.0-1.0).
        llm: Optional language model for RAG answer generation.

    Example:
        >>> pipeline = QdrantHybridSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="transformer architecture explained", top_k=10, alpha=0.6
        ... )
        >>> print(f"Retrieved {len(results['documents'])} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain qdrant section with connection details and
                optional collection_name, alpha settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Configuration Schema:
            qdrant:
              url: "http://localhost:6333"
              api_key: null  # Optional for authenticated instances
              collection_name: "hybrid-collection"
              alpha: 0.5  # Fusion weight
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name"),
        )

        self.collection_name = qdrant_config.get("collection_name")
        self.alpha = qdrant_config.get("alpha", 0.5)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Qdrant hybrid search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search combining dense and sparse embeddings.

        Generates both dense (semantic) and sparse (lexical) embeddings for
        the query, then executes Qdrant's native hybrid search. Optionally
        generates a RAG answer if an LLM was configured.

        Args:
            query: Search query text to embed and search.
            top_k: Maximum number of results to return. Defaults to 10.
            filters: Optional metadata filters as dictionary for pre-filtering.
                Supports Qdrant's filter conditions on payload fields.

        Returns:
            Dictionary containing:
                - documents: List of retrieved Document objects
                - query: Original query string
                - answer: Generated RAG answer (only if LLM configured)

        Raises:
            RuntimeError: If database connection fails during search.

        Sparse Embedding Details:
            Sparse embeddings are generated using TF-IDF vectorization with
            a vocabulary derived from the corpus. Non-zero entries represent
            term importance for keyword matching.
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
        logger.info("Retrieved %d documents from Qdrant", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
