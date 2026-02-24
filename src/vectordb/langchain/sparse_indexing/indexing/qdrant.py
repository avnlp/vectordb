"""Qdrant sparse indexing pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class QdrantSparseIndexingPipeline:
    """Qdrant indexing pipeline for sparse search (LangChain)."""

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = SparseEmbedder()

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name", "sparse_search")

        logger.info("Initialized Qdrant sparse indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline."""
        limit = self.config.get("dataloader", {}).get("limit")
        dl_config = self.config.get("dataloader", {})
        loader = DataloaderCatalog.create(
            dl_config.get("type", "triviaqa"),
            split=dl_config.get("split", "test"),
            limit=limit,
        )
        dataset = loader.load()
        documents = dataset.to_langchain()
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        texts = [doc.page_content for doc in documents]
        sparse_embeddings = self.embedder.embed_documents(texts)
        logger.info("Generated sparse embeddings for %d documents", len(documents))

        # Prepare data for Qdrant with sparse embeddings
        upsert_data = []
        for i, (doc, sparse_emb) in enumerate(zip(documents, sparse_embeddings)):
            upsert_data.append(
                {
                    "text": doc.page_content,
                    "sparse_vector": sparse_emb,
                    "metadata": doc.metadata or {},
                    "doc_id": f"qdrant_{i}",
                }
            )

        num_indexed = self.db.upsert(
            documents=upsert_data,
            embeddings=None,  # No dense embeddings for sparse search
            collection_name=self.collection_name,
        )
        logger.info(
            "Indexed %d documents with sparse embeddings to Qdrant", num_indexed
        )

        return {"documents_indexed": num_indexed}
