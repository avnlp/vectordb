"""Chroma indexing pipeline for contextual compression.

Prepares Chroma vector store for contextual compression search by indexing
documents with dense embeddings. Uses cosine similarity for vector search.

Schema:
    - id: Unique document identifier
    - content: Document text content
    - embedding: Dense vector representation (cosine distance)
    - metadata: JSON-serialized document metadata

Chroma Characteristics:
    - Persistent client stores data on disk
    - Simple schema with automatic embedding handling
    - Cosine distance metric for similarity search

Compression Context:
    Documents indexed here will be retrieved and compressed by ChromaCompressionSearch.
    The indexed documents serve as the candidate pool for compression algorithms.
"""

import chromadb
from haystack import Document

from vectordb.haystack.contextual_compression.indexing.base_indexing import (
    BaseIndexingPipeline,
)


class ChromaIndexingPipeline(BaseIndexingPipeline):
    """Chroma indexing pipeline for contextual compression.

    Loads documents, generates embeddings, and stores in Chroma with simple schema.
    """

    def _connect(self) -> None:
        """Establish connection to Chroma vector database."""
        chroma_config = self.config.get("chroma", {})
        path = chroma_config.get("path", "./chroma_data")
        persist_directory = chroma_config.get("persist_directory", path)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.logger.info("Connected to Chroma at %s", persist_directory)

    def _prepare_collection(self) -> None:
        """Create or get Chroma collection with simple schema."""
        chroma_config = self.config.get("chroma", {})
        collection_name = chroma_config.get("collection_name", "compression")

        # Chroma automatically uses the embedder from our dense_embedder
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.logger.info("Collection '%s' ready", collection_name)

    def _store_documents(self, documents: list[Document]) -> None:
        """Store embedded documents in Chroma.

        Args:
            documents: List of Document objects with embeddings.
        """
        ids = []
        embeddings = []
        documents_list = []
        metadatas = []

        for i, doc in enumerate(documents):
            ids.append(f"{id(doc)}_{i}")
            embeddings.append(doc.embedding)
            documents_list.append(doc.content)

            # Store metadata
            metadata = doc.meta.copy() if doc.meta else {}
            metadatas.append(metadata)

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_list,
                metadatas=metadatas,
            )
            self.logger.debug("Stored %d documents in Chroma", len(documents))
        except Exception as e:
            self.logger.error("Failed to store documents: %s", str(e))
            raise
