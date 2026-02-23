"""Weaviate indexing pipeline for contextual compression.

Prepares Weaviate vector store for contextual compression search with
class-based schema and modular AI integrations.

Schema (Weaviate Class):
    - content: Document text (TEXT type)
    - metadata_json: Serialized metadata (TEXT type)
    - vector: External embedding (provided by pipeline, not Weaviate vectorizer)

Collection Configuration:
    - Vectorizer: None (we provide embeddings manually)
    - Properties: content, metadata_json as TEXT fields
    - Supports both local and cloud (WCS) deployments

Weaviate Characteristics:
    - GraphQL interface for complex queries
    - Modular AI integrations (Q&A, summarization)
    - Schema-first design with strong typing

Compression Context:
    Documents are indexed with external embeddings, retrieved via near_vector
    search, then processed by WeaviateCompressionSearch compression pipeline.
"""

import json

import weaviate
from haystack import Document
from weaviate.classes.config import Configure, Property
from weaviate.classes.config import DataType as WeaviateDataType

from vectordb.haystack.contextual_compression.indexing.base_indexing import (
    BaseIndexingPipeline,
)


class WeaviateIndexingPipeline(BaseIndexingPipeline):
    """Weaviate indexing pipeline for contextual compression.

    Loads documents, generates embeddings, and stores in Weaviate with simple schema.
    """

    def _connect(self) -> None:
        """Establish connection to Weaviate vector database."""
        weaviate_config = self.config.get("weaviate", {})
        url = weaviate_config.get("url", "http://localhost:8080")

        self.client = (
            weaviate.connect_to_local(url=url)
            if "localhost" in url
            else weaviate.connect_to_cloud(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(
                    weaviate_config.get("api_key", "")
                ),
            )
        )

        self.logger.info("Connected to Weaviate at %s", url)

    def _prepare_collection(self) -> None:
        """Create or verify Weaviate collection with simple schema."""
        weaviate_config = self.config.get("weaviate", {})
        collection_name = weaviate_config.get("collection_name", "Compression")

        if self.client.collections.exists(collection_name):
            self.logger.info("Collection '%s' already exists", collection_name)
            self.collection = self.client.collections.get(collection_name)
        else:
            self.collection = self.client.collections.create(
                name=collection_name,
                properties=[
                    Property(
                        name="content",
                        data_type=WeaviateDataType.TEXT,
                    ),
                    Property(
                        name="metadata_json",
                        data_type=WeaviateDataType.TEXT,
                    ),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # We provide embeddings
            )
            self.logger.info("Created collection '%s'", collection_name)

    def _store_documents(self, documents: list[Document]) -> None:
        """Store embedded documents in Weaviate.

        Args:
            documents: List of Document objects with embeddings.
        """
        objects = []
        for doc in documents:
            # Convert metadata to JSON string for storage
            metadata_json = json.dumps(doc.meta) if doc.meta else "{}"

            obj = {
                "content": doc.content,
                "metadata_json": metadata_json,
            }
            objects.append(obj)

        # Batch add objects with embeddings
        try:
            with self.collection.batch.dynamic() as batch:
                for i, obj in enumerate(objects):
                    batch.add_object(
                        properties=obj,
                        vector=documents[i].embedding,
                    )

            self.logger.debug("Stored %d documents in Weaviate", len(documents))
        except Exception as e:
            self.logger.error("Failed to store documents: %s", str(e))
            raise
