"""Qdrant sparse indexing pipeline for keyword/BM25-style search."""

from pathlib import Path
from typing import Any

from haystack.dataclasses import SparseEmbedding

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


def _to_qdrant_sparse(sparse: SparseEmbedding) -> Any:
    """Convert Haystack SparseEmbedding to Qdrant SparseVector format."""
    from qdrant_client.http.models import SparseVector

    return SparseVector(
        indices=list(sparse.indices),
        values=list(sparse.values),
    )


class QdrantSparseIndexingPipeline:
    """Qdrant sparse indexing pipeline using SPLADE embeddings.

    Uses SentenceTransformersSparseDocumentEmbedder for SPLADE embeddings
    and stores them in Qdrant sparse vector format.
    """

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["qdrant"]

        self.db = QdrantVectorDB(
            location=db_config.get("location"),
            collection_name=db_config.get("collection_name"),
            api_key=db_config.get("api_key"),
        )

        self.embedder = EmbedderFactory.create_sparse_document_embedder(self.config)
        self.batch_size = self.config.get("indexing", {}).get("batch_size", 100)

        logger.info(
            f"Initialized QdrantSparseIndexingPipeline with collection: {db_config.get('collection_name')}"
        )

    def create_collection(
        self, dimension: int = 1024, sparse_dimension: int = 30522
    ) -> None:
        """Create collection with sparse vector support.

        Args:
            dimension: Dense vector dimension (dummy for sparse-only).
            sparse_dimension: Sparse vector dimension (typically vocab size for SPLADE).
        """
        self.db.create_collection(
            collection_name=self.db.collection_name,
            dimension=dimension,
            sparse_dimension=sparse_dimension,
        )
        logger.info("Created Qdrant collection with sparse support")

    def run(self) -> dict[str, Any]:
        """Run the indexing pipeline.

        Returns:
            Dict with 'documents_indexed' count.
        """
        # 1. Load documents
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=dl_config.get("limit"),
        )
        dataset = loader.load()
        documents = dataset.to_haystack()
        logger.info(f"Loaded {len(documents)} documents")

        # 2. Embed documents with sparse embeddings
        result = self.embedder.run(documents)
        embedded_docs = result["documents"]
        logger.info(f"Embedded {len(embedded_docs)} documents with sparse vectors")

        # 3. Prepare for Qdrant upsert
        points = []
        for i, doc in enumerate(embedded_docs):
            if doc.sparse_embedding is None:
                continue

            sparse_vector = _to_qdrant_sparse(doc.sparse_embedding)

            point = {
                "id": i,  # Use integer IDs for Qdrant
                "payload": {
                    "content": doc.content or "",
                    "doc_id": doc.id or f"doc_{i}",
                    **(doc.meta or {}),
                },
                "vector": {"sparse": sparse_vector},
            }
            points.append(point)

        # 4. Upsert to Qdrant
        self.db.upsert(
            collection_name=self.db.collection_name,
            points=points,
            batch_size=self.batch_size,
        )

        logger.info(f"Indexed {len(points)} documents to Qdrant")
        return {"documents_indexed": len(points)}
