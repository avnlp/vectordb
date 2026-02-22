"""Qdrant hybrid indexing pipeline (LangChain).

Implements document indexing with both dense and sparse embeddings for
Qdrant's native hybrid search capabilities. Qdrant supports efficient storage
and retrieval of sparse vectors alongside dense vectors in the same payload.

Indexing workflow:
    1. Load documents from configured data source
    2. Generate dense embeddings (semantic meaning via neural networks)
    3. Generate sparse embeddings (lexical keywords via TF-IDF)
    4. Create Qdrant collection with sparse vector configuration
    5. Upsert documents with both embedding types

Qdrant sparse vector format:
    Qdrant accepts sparse vectors as {index: value} dictionaries:
    - index: Integer dimension (position in vocabulary)
    - value: Float weight (TF-IDF score or raw frequency)
    Only non-zero entries are stored, making sparse vectors memory-efficient.

Collection configuration:
    Qdrant collections for hybrid search require:
    - Dense vector field (standard float array)
    - Sparse vector field (dictionary mapping)
    - Payload schema for metadata storage

Upsert payload structure:
    Each document is stored with:
    - id: Unique identifier (collection_name + sequence number)
    - vector: Dense embedding array
    - sparse_vector: Sparse embedding dict {dimension: weight}
    - payload: Document content and custom metadata

Advantages over metadata storage:
    - Native sparse vector indexing for fast retrieval
    - Query-time fusion with alpha weighting
    - Efficient storage through sparse representation
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class QdrantHybridIndexingPipeline:
    """Qdrant hybrid (dense + sparse) indexing pipeline.

    Indexes documents with both dense semantic embeddings and sparse lexical
    embeddings using Qdrant's native sparse vector support.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for generating dense semantic vectors.
        sparse_embedder: Embedder for generating sparse TF-IDF vectors.
        db: QdrantVectorDB instance for database operations.
        collection_name: Target Qdrant collection name.
        dimension: Vector dimension (must match dense embedder output size).

    Example:
        >>> pipeline = QdrantHybridIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents to Qdrant")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain qdrant section with url and optional api_key,
                collection_name, dimension settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Configuration Schema:
            qdrant:
              url: "http://localhost:6333"
              api_key: null
              collection_name: "hybrid-collection"
              dimension: 384

            embedder:
              type: "sentence-transformers"
              model: "all-MiniLM-L6-v2"

            dataloader:
              type: "text"
              source: "data/documents/"
              limit: 10000
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        qdrant_config = self.config["qdrant"]
        url = qdrant_config.get("url", "http://localhost:6333")
        # Parse URL to extract host and port
        url_parsed = url.replace("http://", "").replace("https://", "")
        if ":" in url_parsed:
            host, port_str = url_parsed.split(":", 1)
            port = int(port_str)
        else:
            host = url_parsed
            port = 6333
        self.db = QdrantVectorDB(
            host=host,
            port=port,
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name"),
        )

        self.collection_name = qdrant_config.get("collection_name")
        self.dimension = qdrant_config.get("dimension", 384)

        logger.info("Initialized Qdrant hybrid indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute hybrid indexing pipeline.

        Loads documents, generates both dense and sparse embeddings, creates
        the Qdrant collection with sparse vector support, and upserts all
        documents with hybrid embeddings.

        Returns:
            Dictionary containing:
                - documents_indexed: Count of successfully indexed documents
                - db: Database identifier ("qdrant")
                - collection_name: Name of the target collection

        Raises:
            RuntimeError: If database connection fails or upsert errors occur.
            ValueError: If document loading returns invalid data.

        Sparse Vector Details:
            Sparse embeddings use {dimension_index: weight} format optimized
            for Qdrant's sparse vector field. The SparseEmbedder converts
            text to this format using TF-IDF with a learned vocabulary.
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
            return {"documents_indexed": 0, "db": "qdrant"}

        docs, dense_embeddings = EmbedderHelper.embed_documents(
            self.dense_embedder, documents
        )
        logger.info("Generated %d dense embeddings", len(dense_embeddings))

        texts = [doc.page_content for doc in documents]
        sparse_embeddings = self.sparse_embedder.embed_documents(texts)
        logger.info("Generated %d sparse embeddings", len(sparse_embeddings))

        self.db.create_collection(
            collection_name=self.collection_name,
            vector_size=self.dimension,
        )
        logger.info("Created Qdrant collection: %s", self.collection_name)

        upsert_data = []
        for i, (doc, dense_emb, sparse_emb) in enumerate(
            zip(docs, dense_embeddings, sparse_embeddings)
        ):
            upsert_data.append(
                {
                    "id": f"{self.collection_name}_{i}",
                    "values": dense_emb,
                    "sparse_values": sparse_emb,
                    "metadata": {
                        "text": doc.page_content,
                        **(doc.metadata or {}),
                    },
                }
            )

        num_indexed = self.db.upsert(
            data=upsert_data,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Qdrant", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "db": "qdrant",
            "collection_name": self.collection_name,
        }
