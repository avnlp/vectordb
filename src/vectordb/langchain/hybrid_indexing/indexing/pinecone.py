"""Pinecone hybrid indexing pipeline (LangChain).

Implements document indexing with both dense and sparse embeddings for
Pinecone's native hybrid search capabilities. This pipeline prepares documents
with dual embeddings that enable combined semantic and lexical retrieval.

Indexing workflow:
    1. Load documents from configured data source
    2. Generate dense embeddings (semantic meaning via neural networks)
    3. Generate sparse embeddings (lexical keywords via TF-IDF)
    4. Create or recreate Pinecone index with appropriate dimension
    5. Upsert documents with both embedding types

Pinecone sparse vector format:
    Pinecone accepts sparse vectors as {index: value} dictionaries:
    - index: Integer token ID from the sparse vocabulary
    - value: Float representing term frequency or TF-IDF weight
    Only non-zero entries are transmitted for efficiency.

Upsert payload structure:
    Each document is stored with:
    - id: Unique identifier (index_name + sequence number)
    - values: Dense embedding array (float list)
    - sparse_values: Sparse embedding dict {token_id: weight}
    - metadata: Document content and custom metadata

Index requirements:
    - Dimension must match dense embedding size (e.g., 384, 768, 1536)
    - Metric typically "cosine" for dense embeddings
    - Sparse embeddings don't affect index dimension configuration
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class PineconeHybridIndexingPipeline:
    """Pinecone hybrid (dense + sparse) indexing pipeline.

    Indexes documents with both dense semantic embeddings and sparse lexical
    embeddings to enable Pinecone's native hybrid search functionality.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for generating dense semantic vectors.
        sparse_embedder: Embedder for generating sparse TF-IDF vectors.
        db: PineconeVectorDB instance for database operations.
        index_name: Target Pinecone index name.
        namespace: Namespace within the index for document organization.
        dimension: Vector dimension (must match dense embedder output size).

    Example:
        >>> pipeline = PineconeHybridIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Successfully indexed {result['documents_indexed']} documents")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain pinecone section with api_key, index_name, and
                optional namespace, dimension, metric settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Configuration Schema:
            pinecone:
              api_key: "${PINECONE_API_KEY}"
              index_name: "my-index"
              namespace: "default"
              dimension: 384
              metric: "cosine"
              recreate: false

            embedder:
              type: "sentence-transformers"
              model: "all-MiniLM-L6-v2"

            dataloader:
              type: "text"
              source: "data/documents/"
              limit: 1000
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone hybrid indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute hybrid indexing pipeline.

        Loads documents, generates both dense and sparse embeddings, creates
        the Pinecone index if needed, and upserts all documents with hybrid
        embeddings.

        Returns:
            Dictionary containing:
                - documents_indexed: Count of successfully indexed documents
                - db: Database identifier ("pinecone")
                - index_name: Name of the target index

        Raises:
            RuntimeError: If database connection fails or upsert errors occur.
            ValueError: If document loading returns invalid data.

        Note:
            If recreate=True in config, the index will be deleted and recreated
            before indexing, losing all existing data.
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
            return {"documents_indexed": 0, "db": "pinecone"}

        docs, dense_embeddings = EmbedderHelper.embed_documents(
            self.dense_embedder, documents
        )
        logger.info("Generated %d dense embeddings", len(dense_embeddings))

        texts = [doc.page_content for doc in documents]
        sparse_embeddings = self.sparse_embedder.embed_documents(texts)
        logger.info("Generated %d sparse embeddings", len(sparse_embeddings))

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        upsert_data = []
        for i, (doc, dense_emb, sparse_emb) in enumerate(
            zip(docs, dense_embeddings, sparse_embeddings)
        ):
            upsert_data.append(
                {
                    "id": f"{self.index_name}_{i}",
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
            namespace=self.namespace,
        )
        logger.info("Indexed %d documents to Pinecone index", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "db": "pinecone",
            "index_name": self.index_name,
        }
