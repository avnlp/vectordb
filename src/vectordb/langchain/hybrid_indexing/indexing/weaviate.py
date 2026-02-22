"""Weaviate hybrid indexing pipeline (LangChain).

Implements document indexing with dense embeddings and optional sparse metadata
for Weaviate's hybrid search capabilities. Weaviate uses BM25 for keyword
search combined with dense vector search rather than explicit sparse vectors.

Indexing workflow:
    1. Load documents from configured data source
    2. Generate dense embeddings for semantic search
    3. Generate sparse embeddings (stored in metadata for reference)
    4. Create Weaviate collection with appropriate schema
    5. Upsert documents with dense embeddings

Weaviate hybrid search mechanism:
    Unlike Pinecone/Qdrant that use explicit sparse vectors, Weaviate:
    - Indexes text content for BM25 keyword search at query time
    - Stores dense vectors for semantic similarity
    - Fuses BM25 and vector scores using alpha weighting

Why sparse embeddings are generated:
    Sparse embeddings are generated and can be stored in metadata for:
    - Debugging and analysis
    - Custom reranking implementations
    - Consistency with other hybrid pipelines
    - Future custom hybrid search extensions

Collection schema:
    Weaviate collections are created with:
    - Vector index for dense embeddings (cosine/dot distance)
    - Inverted index for text content (enables BM25)
    - Metadata properties for custom fields
"""

import logging
import uuid
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class WeaviateHybridIndexingPipeline:
    """Weaviate hybrid indexing pipeline.

        Indexes documents with dense embeddings for semantic search. Weaviate's
        native hybrid search combines these dense vectors with BM25 keyword
        matching on the indexed text content.

    Attributes:
            config: Validated configuration dictionary.
            dense_embedder: Embedder for generating dense semantic vectors.
            sparse_embedder: Embedder for generating sparse vectors (metadata only).
            db: WeaviateVectorDB instance for database operations.
            collection_name: Target Weaviate collection/class name.
            dimension: Vector dimension (must match dense embedder output size).

    Example:
        >>> pipeline = WeaviateHybridIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} docs to "
              f"{result['collection_name']}")
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain weaviate section with cluster_url and api_key,
                collection_name, dimension settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Configuration Schema:
            weaviate:
              cluster_url: "http://localhost:8080"
              api_key: null
              collection_name: "Documents"
              dimension: 384

            embedder:
              type: "sentence-transformers"
              model: "all-MiniLM-L6-v2"

            dataloader:
              type: "text"
              source: "data/"
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            cluster_url=weaviate_config.get("cluster_url", "http://localhost:8080"),
            api_key=weaviate_config.get("api_key", ""),
        )

        self.collection_name = weaviate_config.get("collection_name")
        self.dimension = weaviate_config.get("dimension", 384)

        logger.info("Initialized Weaviate hybrid indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute hybrid indexing pipeline.

        Loads documents, generates embeddings, creates the Weaviate collection,
        and upserts all documents with hybrid embeddings.

        Returns:
            Dictionary containing:
                - documents_indexed: Count of successfully indexed documents
                - db: Database identifier ("weaviate")
                - collection_name: Name of the target collection

        Raises:
            RuntimeError: If database connection fails or upsert errors occur.
            ValueError: If document loading returns invalid data.

        Note:
            Sparse embeddings are generated but not required for Weaviate's
            hybrid search, which uses BM25 on text content instead.
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
            return {"documents_indexed": 0, "db": "weaviate"}

        docs, dense_embeddings = EmbedderHelper.embed_documents(
            self.dense_embedder, documents
        )
        logger.info("Generated %d dense embeddings", len(dense_embeddings))

        texts = [doc.page_content for doc in documents]
        self.sparse_embedder.embed_documents(texts)
        logger.info("Generated sparse embeddings (metadata only)")

        self.db.create_collection(
            collection_name=self.collection_name,
        )
        logger.info("Created Weaviate collection: %s", self.collection_name)

        upsert_data = []
        for doc, dense_emb in zip(docs, dense_embeddings):
            upsert_data.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": dense_emb,
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
        logger.info("Indexed %d documents to Weaviate", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "db": "weaviate",
            "collection_name": self.collection_name,
        }
