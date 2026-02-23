"""Pinecone indexing pipeline for contextual compression.

Prepares Pinecone vector store for contextual compression search.
Optimized for managed, serverless vector search with automatic scaling.

Schema:
    - id: Unique document identifier (string)
    - values: Dense embedding vector
    - metadata: content (truncated to 50000 chars), metadata_json

Index Configuration:
    - Serverless architecture (AWS/cloud hosted)
    - Configurable metric: cosine (default), dotproduct, euclidean
    - Auto-scaling based on vector count

Pinecone Characteristics:
    - Fully managed service with no index tuning required
    - Metadata filtering support for hybrid search
    - Content size limit: 50KB per metadata field

Compression Context:
    Documents are indexed with truncated content for metadata efficiency.
    Full content is used during retrieval for compression algorithms.
"""

import json

from haystack import Document
from pinecone import Pinecone, ServerlessSpec

from vectordb.haystack.contextual_compression.indexing.base_indexing import (
    BaseIndexingPipeline,
)


class PineconeIndexingPipeline(BaseIndexingPipeline):
    """Pinecone indexing pipeline for contextual compression.

    Loads documents, generates embeddings, and stores in Pinecone with simple schema.
    """

    def _connect(self) -> None:
        """Establish connection to Pinecone vector database."""
        pinecone_config = self.config.get("pinecone", {})
        api_key = pinecone_config.get("api_key")

        if not api_key:
            raise ValueError("pinecone.api_key not specified in config")

        self.pc = Pinecone(api_key=api_key)
        self.logger.info("Connected to Pinecone")

    def _prepare_collection(self) -> None:
        """Create or verify Pinecone index with simple schema."""
        pinecone_config = self.config.get("pinecone", {})
        index_name = pinecone_config.get("index_name", "compression")
        embedding_dim = self.config.get("embeddings", {}).get("dimension", 384)
        metric = pinecone_config.get("metric", "cosine")

        indexes = self.pc.list_indexes()
        index_exists = any(idx.name == index_name for idx in indexes)

        if index_exists:
            self.logger.info("Index '%s' already exists", index_name)
            self.index = self.pc.Index(index_name)
        else:
            self.pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=pinecone_config.get("cloud", "aws"),
                    region=pinecone_config.get("region", "us-east-1"),
                ),
            )
            self.logger.info(
                "Created index '%s' with dimension %d, metric %s",
                index_name,
                embedding_dim,
                metric,
            )
            self.index = self.pc.Index(index_name)

    def _store_documents(self, documents: list[Document]) -> None:
        """Store embedded documents in Pinecone.

        Args:
            documents: List of Document objects with embeddings.
        """
        # Prepare vectors for upsert
        vectors = []
        for i, doc in enumerate(documents):
            # Convert metadata to JSON string for storage
            metadata_json = json.dumps(doc.meta) if doc.meta else "{}"

            vector = {
                "id": f"{id(doc)}_{i}",  # Unique ID
                "values": doc.embedding,
                "metadata": {
                    "content": doc.content[:50000],  # Limit content size
                    "metadata_json": metadata_json,
                },
            }
            vectors.append(vector)

        # Upsert to Pinecone
        try:
            self.index.upsert(vectors=vectors)
            self.logger.debug("Stored %d documents in Pinecone", len(documents))
        except Exception as e:
            self.logger.error("Failed to store documents: %s", str(e))
            raise
