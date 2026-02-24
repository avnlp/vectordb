"""Pinecone sparse-only indexing pipeline for keyword/BM25-style search."""

from pathlib import Path
from typing import Any

from haystack.dataclasses import SparseEmbedding

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


def _to_pinecone_sparse(sparse: SparseEmbedding) -> dict[str, list]:
    """Convert Haystack SparseEmbedding to Pinecone sparse_values format."""
    return {
        "indices": list(sparse.indices),
        "values": list(sparse.values),
    }


class PineconeSparseIndexingPipeline:
    """Pinecone sparse-only indexing pipeline using SPLADE embeddings.

    Uses SentenceTransformersSparseDocumentEmbedder for SPLADE embeddings
    and stores them in Pinecone's sparse_values format.
    """

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["pinecone"]

        self.db = PineconeVectorDB(
            api_key=db_config.get("api_key"),
            index_name=db_config.get("index_name"),
        )

        self.embedder = EmbedderFactory.create_sparse_document_embedder(self.config)
        self.batch_size = self.config.get("indexing", {}).get("batch_size", 100)
        self.namespace = db_config.get("namespace", "")

        logger.info(
            f"Initialized PineconeSparseIndexingPipeline with index: {db_config.get('index_name')}"
        )

    def create_index(self, dimension: int = 1, metric: str = "dotproduct") -> None:
        """Create sparse-only index.

        Args:
            dimension: Dimension (1 for sparse-only, dummy dense vector).
            metric: Distance metric (dotproduct for sparse).
        """
        self.db.create_index(dimension=dimension, metric=metric)
        logger.info("Created Pinecone sparse index")

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

        # 3. Prepare for Pinecone upsert
        vectors = []
        for i, doc in enumerate(embedded_docs):
            if doc.sparse_embedding is None:
                continue

            sparse_values = _to_pinecone_sparse(doc.sparse_embedding)
            vector_data = {
                "id": doc.id or f"doc_{i}",
                "values": [0.0],  # Dummy dense vector for sparse-only index
                "sparse_values": sparse_values,
                "metadata": {
                    "content": doc.content or "",
                    **(doc.meta or {}),
                },
            }
            vectors.append(vector_data)

        # 4. Upsert to Pinecone
        self.db.upsert(
            data=vectors,
            namespace=self.namespace,
            batch_size=self.batch_size,
            show_progress=True,
        )

        logger.info(f"Indexed {len(vectors)} documents to Pinecone")
        return {"documents_indexed": len(vectors)}
