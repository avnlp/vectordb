"""Pinecone search pipeline with contextual compression.

Retrieves documents from Pinecone managed vector store and applies compression
to filter irrelevant content before returning to LLM.

Retrieval Strategy:
    1. Generate query embedding using configured embedder
    2. Query Pinecone index with configured metric (cosine default)
    3. Extract content and metadata from Pinecone match results
    4. Return documents with pinecone_id and similarity scores

Compression Integration:
    Base class orchestrates compression after retrieval:
    - Retrieves top_k*2 documents for compression pool
    - Compressor filters to final top_k most relevant
    - Supports both reranking and LLM extraction methods

Pinecone-Specific Notes:
    - Fully managed service (no index tuning needed)
    - Content stored in metadata (truncated to 50KB limit)
    - Supports serverless and pod-based deployments

Compression Benefits with Pinecone:
    - Metadata size limits encourage compression necessity
    - High-throughput retrieval supports large initial top_k
    - Compression reduces downstream token costs significantly

Example:
    >>> pipeline = PineconeCompressionSearch("configs/pinecone/popqa/reranking.yaml")
    >>> results = pipeline.run("Who invented the telephone?", top_k=5)
"""

from haystack import Document
from pinecone import Pinecone

from vectordb.haystack.contextual_compression.base import (
    BaseContextualCompressionPipeline,
)


class PineconeCompressionSearch(BaseContextualCompressionPipeline):
    """Pinecone search pipeline with contextual compression.

    Retrieves documents via dense search and applies compression
    (reranking or LLM extraction).
    """

    def _connect(self) -> None:
        """Establish connection to Pinecone."""
        pinecone_config = self.config.get("pinecone", {})
        api_key = pinecone_config.get("api_key")

        if not api_key:
            raise ValueError("pinecone.api_key not specified in config")

        self.pc = Pinecone(api_key=api_key)
        self.logger.info("Connected to Pinecone")

    def _ensure_collection_ready(self) -> None:
        """Verify Pinecone index exists."""
        pinecone_config = self.config.get("pinecone", {})
        index_name = pinecone_config.get("index_name")

        if not index_name:
            raise ValueError("pinecone.index_name not specified in config")

        self.index = self.pc.Index(index_name)
        self.logger.info("Index '%s' ready", index_name)

    def _retrieve_base_results(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Pinecone via dense search.

        Args:
            query: Search query text.
            top_k: Number of results to retrieve.

        Returns:
            List of Document objects with similarity scores in metadata.
        """
        # Embed query
        embedding_result = self.dense_embedder.run(text=query)
        query_embedding = embedding_result["embedding"]

        # Search
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
        )

        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            doc = Document(
                content=metadata.pop("content", ""),
                meta={
                    "score": match.get("score", 0.0),
                    "pinecone_id": match.get("id"),
                    **metadata,
                },
            )
            documents.append(doc)

        self.logger.debug("Retrieved %d documents from Pinecone", len(documents))
        return documents
