"""Weaviate search pipeline with contextual compression.

Retrieves documents from Weaviate vector store using near_vector search and
applies compression to return only the most relevant content chunks.

Retrieval Strategy:
    1. Generate query embedding with dense embedder
    2. Execute near_vector search with distance metric
    3. Convert Weaviate distance to similarity score (1 - distance)
    4. Deserialize metadata JSON for document attributes

Compression Integration:
    Retrieved documents are processed by compression pipeline:
    - BaseContextualCompressionPipeline.run() retrieves top_k*2 candidates
    - Compressor (reranker or LLM) filters to optimal top_k
    - Returns compressed documents with relevance scores

Weaviate-Specific Notes:
    - Schema-first design with class-based collections
    - GraphQL query interface (wrapped by Python client)
    - Supports both local Docker and Weaviate Cloud Service (WCS)

Compression Benefits with Weaviate:
    - Vector search returns candidates with distance metrics
    - Compression fine-tunes relevance beyond vector similarity
    - Reduces context window usage for LLM generation tasks

Example:
    >>> pipeline = WeaviateCompressionSearch(
    ...     "configs/weaviate/earnings_calls/reranking.yaml"
    ... )
    >>> results = pipeline.run("Q3 revenue growth", top_k=5)
"""

import ast

import weaviate
from haystack import Document

from vectordb.haystack.contextual_compression.base import (
    BaseContextualCompressionPipeline,
)


class WeaviateCompressionSearch(BaseContextualCompressionPipeline):
    """Weaviate search pipeline with contextual compression.

    Retrieves documents via dense search and applies compression
    (reranking or LLM extraction).
    """

    def _connect(self) -> None:
        """Establish connection to Weaviate."""
        weaviate_config = self.config.get("weaviate", {})
        url = weaviate_config.get("url", "http://localhost:8080")

        self.client = weaviate.connect_to_local(host=url.split("://")[-1].split(":")[0])
        self.logger.info("Connected to Weaviate at %s", url)

    def _ensure_collection_ready(self) -> None:
        """Verify Weaviate collection exists."""
        weaviate_config = self.config.get("weaviate", {})
        collection_name = weaviate_config.get("collection_name", "Compression")

        try:
            self.client.collections.get(collection_name)
            self.collection_name = collection_name
            self.logger.info("Collection '%s' ready", collection_name)
        except Exception as e:
            self.logger.error("Collection '%s' not found: %s", collection_name, str(e))
            raise

    def _retrieve_base_results(self, query: str, top_k: int) -> list[Document]:
        """Retrieve documents from Weaviate via dense search.

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
        collection = self.client.collections.get(self.collection_name)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True),
        )

        documents = []
        for obj in results.objects:
            properties = obj.properties or {}
            content = properties.pop("content", "")
            metadata = properties.pop("metadata", "{}")
            doc = Document(
                content=content,
                meta={
                    "score": 1 - (obj.metadata.distance or 0),
                    "weaviate_distance": obj.metadata.distance,
                    **(ast.literal_eval(metadata) if metadata else {}),
                    **properties,
                },
            )
            documents.append(doc)

        self.logger.debug("Retrieved %d documents from Weaviate", len(documents))
        return documents
