"""Pinecone hybrid search pipeline (LangChain).

Implements hybrid search using Pinecone's native support for separate dense
and sparse vector embeddings. Pinecone fuses results using a configurable
alpha parameter that controls the weighting between semantic (dense) and
lexical (sparse) relevance scores.

The hybrid search mechanism:
    1. Query is embedded using both dense and sparse embedders
    2. Dense embedding captures semantic query intent
    3. Sparse embedding captures exact term matches (TF-IDF style)
    4. Pinecone fuses results with alpha * dense_score + (1-alpha) * sparse_score
    5. Results are ranked by the fused score

Alpha parameter (0.0 to 1.0):
    - 1.0: Pure dense/semantic search
    - 0.0: Pure sparse/keyword search
    - 0.5: Balanced hybrid (default)

Note:
    Pinecone requires separate sparse_values field for sparse embeddings,
    distinct from the standard values field used for dense vectors.
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    HaystackToLangchainConverter,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class PineconeHybridSearchPipeline:
    """Pinecone hybrid (dense + sparse) search pipeline.

    Combines dense semantic embeddings with sparse lexical embeddings for
    enhanced retrieval. Uses Pinecone's native hybrid search fusion with
    configurable alpha weighting.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for semantic vector generation.
        sparse_embedder: Embedder for sparse lexical vectors.
        db: PineconeVectorDB instance for vector operations.
        index_name: Target Pinecone index name.
        namespace: Namespace within the index.
        alpha: Fusion weight (0.0=sparse only, 1.0=dense only, 0.5=hybrid).
        llm: Optional language model for RAG answer generation.

    Example:
        >>> pipeline = PineconeHybridSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="neural network architectures", top_k=10, alpha=0.7
        ... )
        >>> print(f"Found {len(results['documents'])} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain pinecone section with api_key and optional
                index_name, namespace, alpha settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            The embedder configuration is read from config["embedder"].
            The LLM configuration is optional and read from config["llm"].
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.alpha = pinecone_config.get("alpha", 0.5)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Pinecone hybrid search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search combining dense and sparse embeddings.

        Generates both dense (semantic) and sparse (lexical) embeddings for
        the query, then executes Pinecone's native hybrid search. Optionally
        generates a RAG answer if an LLM was configured.

        Args:
            query: Search query text to embed and search.
            top_k: Maximum number of results to return. Defaults to 10.
            filters: Optional metadata filters as dictionary for pre-filtering.
                Example: {"category": "technology", "year": 2024}

        Returns:
            Dictionary containing:
                - documents: List of retrieved Document objects
                - query: Original query string
                - answer: Generated RAG answer (only if LLM configured)

        Raises:
            RuntimeError: If database connection fails during search.

        Note:
            Sparse embeddings use TF-IDF style bag-of-words representation
            where non-zero dimensions correspond to term frequencies.
        """
        dense_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        sparse_embedding = self.sparse_embedder.embed_query(query)
        logger.info("Generated hybrid embeddings for query: %s...", query[:50])

        documents = self.db.hybrid_search(
            query_embedding=dense_embedding,
            query_sparse_embedding=sparse_embedding,
            top_k=top_k,
            filter=filters,
            namespace=self.namespace,
            alpha=self.alpha,
        )
        documents = HaystackToLangchainConverter.convert(documents)
        logger.info("Retrieved %d documents from Pinecone", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
