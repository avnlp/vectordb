"""Pinecone sparse indexing pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class PineconeSparseIndexingPipeline:
    """Pinecone indexing pipeline for sparse search (LangChain).

    Loads documents, generates sparse embeddings, creates index, and indexes.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = SparseEmbedder()

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

        logger.info("Initialized Pinecone sparse indexing pipeline (LangChain)")

    def run(self) -> dict[str, Any]:
        """Execute indexing pipeline.

        Returns:
            Dict with 'documents_indexed' count.
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
        logger.info("Loaded %d documents", len(documents))

        if not documents:
            logger.warning("No documents to index")
            return {"documents_indexed": 0}

        texts = [doc.page_content for doc in documents]
        sparse_embeddings = self.embedder.embed_documents(texts)
        logger.info("Generated sparse embeddings for %d documents", len(documents))

        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        # Upsert with sparse embeddings only
        upsert_data = []
        for i, (doc, sparse_emb) in enumerate(zip(documents, sparse_embeddings)):
            upsert_data.append(
                {
                    "id": f"{self.index_name}_{i}",
                    "values": [0.0] * self.dimension,  # Placeholder dense vector
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
        logger.info(
            "Indexed %d documents with sparse embeddings to Pinecone", num_indexed
        )

        return {"documents_indexed": num_indexed}
