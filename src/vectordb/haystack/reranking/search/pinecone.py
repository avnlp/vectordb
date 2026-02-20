"""Pinecone reranking search pipeline.

This module provides a search pipeline combining Pinecone's managed vector
search service with cross-encoder reranking. Pinecone's serverless architecture
provides scalable dense retrieval, while reranking improves result precision.

Reranking Strategy:
    1. Embed query using bi-encoder for fast vector similarity
    2. Retrieve 3x candidates from Pinecone (top_k * 3)
    3. Apply cross-encoder reranking for final ordering
    4. Return top_k most relevant documents

Cross-Encoder Benefits:
    Unlike bi-encoders that encode query and document separately, cross-encoders
    process them jointly, capturing interactions like negation, coreference,
    and semantic relationships that vector similarity might miss.

Pinecone Integration:
    - Serverless or pod-based deployment options
    - Metadata filtering support for pre-filtering candidates
    - Namespace isolation for multi-tenant scenarios
    - Automatic scaling with usage
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory, RerankerFactory


logger = logging.getLogger(__name__)


class PineconeRerankingSearchPipeline:
    """Pinecone search pipeline with cross-encoder reranking.

    Implements two-stage retrieval: fast Pinecone vector search followed by
    precise cross-encoder reranking. The cross-encoder scores each query-
    document pair to capture semantic nuances missed by vector similarity.

    Attributes:
        config: Pipeline configuration dict with pinecone, embedder, reranker settings.
        embedder: Bi-encoder for query embedding and document retrieval.
        reranker: Cross-encoder for final document scoring and reordering.
        db: PineconeVectorDB instance for vector storage and search.
        index_name: Name of the Pinecone index to query.
        namespace: Optional namespace for data isolation.

    Example:
        >>> pipeline = PineconeRerankingSearchPipeline("pinecone_config.yaml")
        >>> results = pipeline.search(
        ...     query="neural network optimization techniques",
        ...     top_k=5,
        ...     filters={"category": "research"},
        ... )
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file containing:
                - pinecone: API key, index_name, namespace, metric
                - embedder: Provider, model, dimensions, batch_size
                - reranker: Provider, model, top_k

        Raises:
            ValueError: If required config sections are missing or invalid.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        self.reranker = RerankerFactory.create(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        logger.info("Initialized Pinecone reranking search pipeline")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute reranking search with two-stage retrieval.

        Stage 1 - Dense Retrieval:
            Convert query to embedding using bi-encoder, then retrieve
            top_k*3 candidates from Pinecone using vector similarity.

        Stage 2 - Cross-Encoder Reranking:
            Score each candidate with cross-encoder for precise relevance
            assessment, then return top_k results.

        Args:
            query: Search query text to find relevant documents for.
            top_k: Number of final results to return after reranking.
                The pipeline retrieves 3x this amount for reranking.
            filters: Optional metadata filters for Pinecone's structured filtering.
                Supports equality, range, and logical operators.

        Returns:
            Dict with 'query' (original query string) and 'documents' (list of
            reranked Document objects sorted by relevance score, highest first).

        Raises:
            RuntimeError: If embedder or reranker fails during execution.
        """
        # Stage 1: Embed query using bi-encoder for vector search
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Retrieve more candidates than needed (3x) to give reranker options
        retrieval_top_k = top_k * 3
        base_docs = self.db.query(
            query_embedding=query_embedding,
            top_k=retrieval_top_k,
            namespace=self.namespace,
            filter=filters if filters else None,
        )
        logger.info("Retrieved %d base documents", len(base_docs))

        if not base_docs:
            logger.warning("No base documents retrieved")
            return {"query": query, "documents": []}

        # Stage 2: Apply cross-encoder reranking for precision improvement
        # Cross-encoder jointly encodes query+document for fine-grained scoring
        reranked_result = self.reranker.run(query=query, documents=base_docs)
        reranked_docs = reranked_result.get("documents", [])[:top_k]

        logger.info("Reranked to %d documents", len(reranked_docs))
        return {"query": query, "documents": reranked_docs}

    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Alias for search() for backward compatibility with Haystack pipelines."""
        return self.search(query, top_k)
