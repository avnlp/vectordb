"""Chroma hybrid indexing pipeline (LangChain).

Implements document indexing with dense embeddings and sparse metadata storage
for Chroma. Chroma has limited native hybrid search support, so this pipeline
focuses on dense vector indexing while storing sparse embeddings in metadata
for potential custom reranking.

Indexing workflow:
    1. Load documents from configured data source
    2. Generate dense embeddings for primary vector search
    3. Generate sparse embeddings and serialize to metadata
    4. Create Chroma collection
    5. Upsert documents with dense vectors and sparse metadata

Chroma sparse embedding storage:
    Since Chroma doesn't support native sparse vector fields, sparse embeddings
    are stored as JSON-serialized strings in document metadata:
    metadata["sparse_embedding"] = str(sparse_dict)

    This allows:
    - Custom post-search reranking using sparse similarity
    - Debugging and analysis of term distributions
    - Future hybrid search implementation in application layer

Limitations:
    - Sparse embeddings cannot be used in Chroma's native search
    - Custom reranking requires fetching documents first, then rescore
    - Metadata storage increases memory usage per document

Use cases:
    - Local development with potential hybrid extensions
    - Semantic-first search with sparse fallback
    - Prototyping before migrating to full hybrid databases
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class ChromaHybridIndexingPipeline:
    """Chroma hybrid indexing pipeline with sparse metadata storage.

    Indexes documents with dense embeddings for primary search and stores
    sparse embeddings in metadata for custom reranking or analysis.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for generating dense semantic vectors.
        sparse_embedder: Embedder for generating sparse TF-IDF vectors.
        db: ChromaVectorDB instance for database operations.
        collection_name: Target Chroma collection name.

    Example:
        >>> pipeline = ChromaHybridIndexingPipeline("config.yaml")
        >>> result = pipeline.run()
        >>> print(f"Indexed {result['documents_indexed']} documents to Chroma")

    Note:
        Sparse embeddings are stored in metadata as strings since Chroma
        doesn't have native sparse vector support. Use for custom reranking
        or migrate to Pinecone/Qdrant for native hybrid search.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain chroma section with connection details and optional
                collection_name setting.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Configuration Schema:
            chroma:
              host: "localhost"
              port: 8000
              collection_name: "my-collection"

            embedder:
              type: "sentence-transformers"
              model: "all-MiniLM-L6-v2"

            dataloader:
              type: "text"
              source: "data/documents/"
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            host=chroma_config.get("host", "localhost"),
            port=chroma_config.get("port", 8000),
            collection_name=chroma_config.get("collection_name"),
        )

        self.collection_name = chroma_config.get("collection_name")

        logger.info("Initialized Chroma hybrid indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute hybrid indexing pipeline.

                Loads documents, generates both dense and sparse embeddings, creates
                the Chroma collection, and upserts documents with embeddings and
                sparse metadata.

        Returns:
                    Dictionary containing:
                        - documents_indexed: Count of successfully indexed documents
                        - db: Database identifier ("chroma")
                        - collection_name: Name of the target collection

        Raises:
                    RuntimeError: If database connection fails or upsert errors occur.
                    ValueError: If document loading returns invalid data.

        Sparse Storage:
            Sparse embeddings are converted to string representation and stored
            in metadata["sparse_embedding"]. Deserialize with eval() or
            ast.literal_eval() for custom reranking:
            sparse_dict = eval(metadata["sparse_embedding"])
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
            return {"documents_indexed": 0, "db": "chroma"}

        docs, dense_embeddings = EmbedderHelper.embed_documents(
            self.dense_embedder, documents
        )
        logger.info("Generated %d dense embeddings", len(dense_embeddings))

        texts = [doc.page_content for doc in documents]
        sparse_embeddings = self.sparse_embedder.embed_documents(texts)
        logger.info("Generated %d sparse embeddings", len(sparse_embeddings))

        self.db.create_collection(
            collection_name=self.collection_name,
            dimension=len(dense_embeddings[0]) if dense_embeddings else 384,
        )
        logger.info("Created Chroma collection: %s", self.collection_name)

        upsert_data = []
        for i, (doc, dense_emb, sparse_emb) in enumerate(
            zip(docs, dense_embeddings, sparse_embeddings)
        ):
            upsert_data.append(
                {
                    "id": f"{self.collection_name}_{i}",
                    "values": dense_emb,
                    "metadata": {
                        "text": doc.page_content,
                        "sparse_embedding": str(sparse_emb),
                        **(doc.metadata or {}),
                    },
                }
            )

        num_indexed = self.db.upsert(
            data=upsert_data,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {
            "documents_indexed": num_indexed,
            "db": "chroma",
            "collection_name": self.collection_name,
        }
