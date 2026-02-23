"""Qdrant indexing pipeline for contextual compression.

Prepares Qdrant vector store for contextual compression search with
payload-based storage and flexible filtering capabilities.

Schema:
    - id: Unique numeric identifier (hashed from doc reference)
    - vector: Dense embedding with cosine distance metric
    - payload: content (full text), metadata_json (document metadata)

Collection Configuration:
    - Distance: Cosine similarity
    - Vector params: Configurable dimension (default: 384)
    - On-disk storage support for large collections

Qdrant Characteristics:
    - Open-source with REST/gRPC APIs
    - Payload filtering with complex query DSL
    - Efficient sparse vector support (for hybrid search)

Compression Context:
    Documents stored here are retrieved by QdrantCompressionSearch using
    vector similarity, then compressed via reranking or LLM extraction.
"""

import json

from haystack import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from vectordb.haystack.contextual_compression.indexing.base_indexing import (
    BaseIndexingPipeline,
)


class QdrantIndexingPipeline(BaseIndexingPipeline):
    """Qdrant indexing pipeline for contextual compression.

    Loads documents, generates embeddings, and stores in Qdrant with simple schema.
    """

    def _connect(self) -> None:
        """Establish connection to Qdrant vector database."""
        qdrant_config = self.config.get("qdrant", {})
        url = qdrant_config.get("url", "http://localhost:6333")
        api_key = qdrant_config.get("api_key")

        kwargs = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key

        self.client = QdrantClient(**kwargs)
        self.logger.info("Connected to Qdrant at %s", url)

    def _prepare_collection(self) -> None:
        """Create or verify Qdrant collection with simple schema."""
        qdrant_config = self.config.get("qdrant", {})
        collection_name = qdrant_config.get("collection_name", "compression")
        embedding_dim = self.config.get("embeddings", {}).get("dimension", 384)

        try:
            self.client.get_collection(collection_name)
            self.logger.info("Collection '%s' already exists", collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            self.logger.info(
                "Created collection '%s' with embedding dim %d",
                collection_name,
                embedding_dim,
            )

        self.collection_name = collection_name

    def _store_documents(self, documents: list[Document]) -> None:
        """Store embedded documents in Qdrant.

        Args:
            documents: List of Document objects with embeddings.
        """
        points = []
        for i, doc in enumerate(documents):
            # Convert metadata to JSON string for storage
            metadata_json = json.dumps(doc.meta) if doc.meta else "{}"

            point = PointStruct(
                id=abs(hash(f"{id(doc)}_{i}")) % (10**8),  # Unique numeric ID
                vector=doc.embedding,
                payload={
                    "content": doc.content,
                    "metadata_json": metadata_json,
                },
            )
            points.append(point)

        # Upsert to Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            self.logger.debug("Stored %d documents in Qdrant", len(documents))
        except Exception as e:
            self.logger.error("Failed to store documents: %s", str(e))
            raise
