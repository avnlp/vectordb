"""Milvus hybrid indexing pipeline (LangChain).
Milvus's native hybrid search capabilities. Milvus 2.3+ supports sparse
vectors as a separate field type, enabling efficient hybrid retrieval.
    1. Load documents from configured data source
    2. Generate dense embeddings (semantic meaning via neural networks)
    3. Generate sparse embeddings (lexical keywords via TF-IDF)
    4. Create Milvus collection with sparse vector field enabled
    5. Insert documents as Haystack Document objects with both embedding types
Milvus sparse vector format:
    Milvus accepts sparse vectors as dictionaries {index: value}:
    - index: Integer dimension (token position in vocabulary)
    - value: Float weight (typically normalized TF-IDF)
    Sparse vectors are stored efficiently using inverted index structure.
    Milvus collections require separate fields for:
    - Primary key (auto-generated INT64 ID)
    - Dense vector field (float array, indexed via HNSW)
    - Sparse vector field (dict, indexed via inverted index)
    - Content field (VARCHAR for document text)
    - Metadata field (JSON for custom properties)

Document structure:
    Documents are inserted as Haystack Document objects containing:
    - content: Document text string
    - embedding: Dense vector (list of floats)
    - sparse_embedding: Sparse vector {token_id: weight}
    - meta: Additional custom metadata fields

Note: The collection uses auto_id=True, so document IDs are generated
automatically by Milvus (INT64 primary keys).
GPU acceleration:
    Milvus supports GPU indexing for dense vectors (CAGRA index),
    while sparse vectors use CPU-based inverted indexes.
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class MilvusHybridIndexingPipeline:
    """Milvus hybrid (dense + sparse) indexing pipeline.

    Indexes documents with both dense semantic embeddings and sparse lexical
    embeddings using Milvus's native sparse vector field type.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for generating dense semantic vectors.
        sparse_embedder: Embedder for generating sparse TF-IDF vectors.
        db: MilvusVectorDB instance for database operations.
        collection_name: Target Milvus collection name.
        dimension: Vector dimension (must match dense embedder output size).

    Example:
        >>> pipeline = MilvusHybridIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents to Milvus")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain milvus section with connection details and optional
                collection_name, dimension settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Configuration Schema:
            milvus:
              host: "localhost"
              port: 19530
              collection_name: "hybrid-collection"
              dimension: 384

            embedder:
              type: "sentence-transformers"
              model: "all-MiniLM-L6-v2"

            dataloader:
              type: "text"
              source: "data/documents/"
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
            collection_name=milvus_config.get("collection_name"),
        )

        self.collection_name = milvus_config.get("collection_name")
        self.dimension = milvus_config.get("dimension", 384)

        logger.info("Initialized Milvus hybrid indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute hybrid indexing pipeline.

        Loads documents, generates both dense and sparse embeddings, creates
        the Milvus collection with sparse vector field, and inserts all
        documents with hybrid embeddings.

        Returns:
            Dictionary containing:
                - documents_indexed: Count of successfully indexed documents
                - db: Database identifier ("milvus")
                - collection_name: Name of the target collection

        Raises:
            RuntimeError: If database connection fails or insert errors occur.
            ValueError: If document loading returns invalid data.

        Sparse Vector Details:
            Sparse embeddings use {dimension: weight} format where dimensions
            are token positions in the TF-IDF vocabulary. Milvus stores these
            in a dedicated sparse vector field with inverted index for fast
            retrieval during hybrid search.
        """
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_langchain()
        logger.info("Loaded %d documents for indexing", len(documents))

        if not documents:
            logger.warning("No documents loaded; indexing skipped")
            return {"documents_indexed": 0, "db": "milvus"}

        docs, dense_embeddings = EmbedderHelper.embed_documents(
            self.dense_embedder, documents
        )
        logger.info("Generated %d dense embeddings", len(dense_embeddings))

        texts = [doc.page_content for doc in documents]
        sparse_embeddings = self.sparse_embedder.embed_documents(texts)
        logger.info("Generated %d sparse embeddings", len(sparse_embeddings))

        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            use_sparse=True,
        )
        logger.info("Created Milvus collection: %s", self.collection_name)
        # Convert to Haystack Documents for insert_documents
        # Note: auto_id=True on collection, so we don't provide explicit IDs
        from haystack.dataclasses import Document

        from vectordb.utils.sparse import to_milvus_sparse

        documents = []
        for doc, dense_emb, sparse_emb in zip(docs, dense_embeddings, sparse_embeddings):
            haystack_doc = Document(
                content=doc.page_content,
                embedding=dense_emb,
                meta={
                    **(doc.metadata or {}),
                },
            )
            # Attach sparse embedding as attribute (insert_documents handles it)
            haystack_doc.sparse_embedding = to_milvus_sparse(sparse_emb)
            documents.append(haystack_doc)

        num_indexed = self.db.insert_documents(
            documents=documents,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Milvus", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "db": "milvus",
            "collection_name": self.collection_name,
        }
