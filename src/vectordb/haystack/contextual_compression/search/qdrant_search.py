"""Qdrant search pipeline with contextual compression.

Retrieves documents from Qdrant vector store and applies contextual compression
to optimize token usage for LLM context windows.

Retrieval Strategy:
    1. Embed query using dense embedder (e.g., Qwen3-Embedding-0.6B)
    2. Search Qdrant collection using cosine similarity
    3. Extract content and payload metadata from search results
    4. Deserialize JSON metadata for document enrichment

Compression Integration:
    Documents flow through BaseContextualCompressionPipeline:
    - Initial retrieval: top_k*2 candidates
    - Compression: reranking (cross-encoder/Cohere/Voyage) or LLM extraction
    - Final output: top_k filtered documents

Qdrant-Specific Notes:
    - Open-source with flexible deployment (local/cloud)
    - Payload-based storage with complex filtering DSL
    - Supports hybrid search (dense + sparse vectors)

Compression Benefits with Qdrant:
    - Payload filtering enables pre-filtering before compression
    - Efficient HNSW indexing supports fast retrieval of large candidate sets
    - Compression reduces tokens while preserving query relevance

Example:
    >>> pipeline = QdrantCompressionSearch(
    ...     "configs/qdrant/factscore/llm_extraction.yaml"
    ... )
    >>> results = pipeline.run("Fact-check this claim", top_k=3)
"""

import ast
import contextlib
import json

from haystack import Document
from qdrant_client import QdrantClient

from vectordb.haystack.contextual_compression.base import (
    BaseContextualCompressionPipeline,
)


class QdrantCompressionSearch(BaseContextualCompressionPipeline):
    """Qdrant search pipeline with contextual compression.

    Retrieves documents via dense search and applies compression
    (reranking or LLM extraction).
    """

    def _connect(self) -> None:
        """Establish connection to Qdrant."""
        qdrant_config = self.config.get("qdrant", {})
        url = qdrant_config.get("url", "http://localhost:6333")

        self.client = QdrantClient(url=url)
        self.logger.info("Connected to Qdrant at %s", url)

    def _ensure_collection_ready(self) -> None:
        """Verify Qdrant collection exists."""
        qdrant_config = self.config.get("qdrant", {})
        collection_name = qdrant_config.get("collection_name", "compression")

        try:
            self.client.get_collection(collection_name)
            self.collection_name = collection_name
            self.logger.info("Collection '%s' exists", collection_name)
        except Exception as e:
            self.logger.error("Collection '%s' not found: %s", collection_name, str(e))
            raise

    def _retrieve_base_results(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Qdrant via dense search.

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
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        documents = []
        for point in results:
            payload = point.payload or {}
            content = payload.pop("content", "")

            # Handle metadata - try metadata_json first, then metadata field
            metadata = {}
            metadata_str = payload.pop("metadata_json", None) or payload.pop(
                "metadata", None
            )
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python dict strings
                    with contextlib.suppress(ValueError, SyntaxError):
                        metadata = ast.literal_eval(metadata_str)

            doc = Document(
                content=content,
                meta={
                    "score": point.score,
                    "qdrant_id": point.id,
                    **metadata,
                    **payload,
                },
            )
            documents.append(doc)

        self.logger.debug("Retrieved %d documents from Qdrant", len(documents))
        return documents
