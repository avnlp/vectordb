"""Milvus sparse indexing pipeline for keyword/BM25-style search."""

from pathlib import Path
from typing import Any

from haystack.dataclasses import SparseEmbedding

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


def _to_milvus_sparse(sparse: SparseEmbedding) -> dict[int, float]:
    """Convert Haystack SparseEmbedding to Milvus sparse format."""
    return dict(zip(sparse.indices, sparse.values))


class MilvusSparseIndexingPipeline:
    """Milvus sparse indexing pipeline using SPLADE embeddings.

    Uses SentenceTransformersSparseDocumentEmbedder for SPLADE embeddings
    and stores them in Milvus sparse format.
    """

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["milvus"]

        self.db = MilvusVectorDB(
            connection_args=db_config.get("connection_args", {}),
            collection_name=db_config.get("collection_name"),
        )

        self.embedder = EmbedderFactory.create_sparse_document_embedder(self.config)
        self.batch_size = self.config.get("indexing", {}).get("batch_size", 100)

        logger.info(
            f"Initialized MilvusSparseIndexingPipeline with collection: {db_config.get('collection_name')}"
        )

    def create_collection(
        self, dimension: int = 1024, enable_sparse: bool = True
    ) -> None:
        """Create collection with sparse vector support.

        Args:
            dimension: Dense vector dimension (dummy for sparse-only).
            enable_sparse: Whether to enable sparse vector support.
        """
        self.db.create_collection(
            dimension=dimension,
            enable_sparse=enable_sparse,
            sparse_index_type="SPARSE_INVERTED_INDEX",
            sparse_metric_type="IP",
        )
        logger.info("Created Milvus collection with sparse support")

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

        # 3. Prepare for Milvus insertion
        texts = []
        sparse_vectors = []
        metadatas = []

        for i, doc in enumerate(embedded_docs):
            if doc.sparse_embedding is None:
                continue

            texts.append(doc.content or "")
            sparse_vectors.append(_to_milvus_sparse(doc.sparse_embedding))
            metadatas.append(
                {
                    "doc_id": doc.id or f"doc_{i}",
                    **(doc.meta or {}),
                }
            )

        # 4. Insert to Milvus
        self.db.insert_texts(
            texts=texts,
            sparse_vectors=sparse_vectors,
            metadatas=metadatas,
            batch_size=self.batch_size,
        )

        logger.info(f"Indexed {len(texts)} documents to Milvus")
        return {"documents_indexed": len(texts)}
