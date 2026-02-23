"""Milvus indexing pipeline for contextual compression.

Prepares Milvus vector store for contextual compression search with optimized
indexing for large-scale document collections.

Schema:
    - id: Auto-generated primary key (INT64)
    - content: Document text (VARCHAR, max 65535 chars)
    - embedding: Float vector with configurable dimension
    - metadata: JSON-serialized metadata (VARCHAR, max 65535 chars)

Index Configuration:
    - Type: IVF_FLAT (Inverted File with flat quantization)
    - Metric: IP (Inner Product)
    - nlist: 128 (number of clusters for partitioning)

Milvus Characteristics:
    - Distributed architecture supports horizontal scaling
    - IVF_FLAT index balances recall and query speed
    - Inner product metric for maximum inner product search (MIPS)

Compression Context:
    Indexed documents are retrieved via MilvusCompressionSearch and filtered
    through compression algorithms (reranking or LLM extraction).
"""

from haystack import Document
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from vectordb.haystack.contextual_compression.indexing.base_indexing import (
    BaseIndexingPipeline,
)


class MilvusIndexingPipeline(BaseIndexingPipeline):
    """Milvus indexing pipeline for contextual compression.

    Loads documents, generates embeddings, and stores in Milvus with simple schema.
    """

    def _connect(self) -> None:
        """Establish connection to Milvus vector database."""
        milvus_config = self.config.get("milvus", {})
        host = milvus_config.get("host", "localhost")
        port = milvus_config.get("port", 19530)

        connections.connect(alias="default", host=host, port=int(port))
        self.logger.info("Connected to Milvus at %s:%s", host, port)

    def _prepare_collection(self) -> None:
        """Create or verify Milvus collection with simple schema."""
        milvus_config = self.config.get("milvus", {})
        collection_name = milvus_config.get("collection_name", "compression")
        embedding_dim = self.config.get("embeddings", {}).get("dimension", 384)

        # Drop existing collection if configured
        if utility.has_collection(collection_name):
            drop_existing = milvus_config.get("drop_existing", False)
            if drop_existing:
                utility.drop_collection(collection_name)
                self.logger.info("Dropped existing collection: %s", collection_name)
            else:
                self.logger.info("Collection '%s' already exists", collection_name)
                self.collection_name = collection_name
                return

        # Define schema: id, content, embedding, metadata (as JSON string)
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_dim,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.VARCHAR,
                max_length=65535,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"Contextual compression collection for {collection_name}",
        )

        collection = Collection(
            name=collection_name,
            schema=schema,
            using="default",
        )

        collection.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            },
        )

        self.logger.info(
            "Created collection '%s' with embedding dim %d",
            collection_name,
            embedding_dim,
        )
        self.collection_name = collection_name

    def _store_documents(self, documents: list[Document]) -> None:
        """Store embedded documents in Milvus.

        Args:
            documents: List of Document objects with embeddings.
        """
        import json

        collection = Collection(self.collection_name)

        # Prepare data
        contents = []
        embeddings = []
        metadata_strs = []

        for doc in documents:
            contents.append(doc.content)
            embeddings.append(doc.embedding)
            # Store metadata as JSON string
            metadata_json = json.dumps(doc.meta) if doc.meta else "{}"
            metadata_strs.append(metadata_json)

        # Insert into Milvus
        try:
            collection.insert(
                data=[
                    contents,
                    embeddings,
                    metadata_strs,
                ]
            )
            self.logger.debug("Stored %d documents in Milvus", len(documents))
        except Exception as e:
            self.logger.error("Failed to store documents: %s", str(e))
            raise
