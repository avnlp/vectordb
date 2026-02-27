"""Chroma diversity filtering indexing pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
)


logger = logging.getLogger(__name__)


class ChromaDiversityFilteringIndexingPipeline:
    """Chroma indexing pipeline for diversity filtering (LangChain).

    Loads documents, generates embeddings, creates index, and indexes.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_dir=chroma_config.get("persist_dir"),
        )

        self.collection_name = chroma_config.get("collection_name")

        logger.info(
            "Initialized Chroma diversity filtering indexing pipeline (LangChain)"
        )

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

        docs, embeddings = EmbedderHelper.embed_documents(self.embedder, documents)
        logger.info("Generated embeddings for %d documents", len(docs))

        recreate = self.config.get("chroma", {}).get("recreate", False)
        if recreate:
            self.db.delete_collection(name=self.collection_name)
            self.db.create_collection(
                name=self.collection_name,
                get_or_create=False,
            )
        else:
            self.db.create_collection(
                name=self.collection_name,
                get_or_create=True,
            )

        # Upsert documents
        num_indexed = self.db.upsert(
            documents=docs,
            embeddings=embeddings,
            collection_name=self.collection_name,
        )
        logger.info("Indexed %d documents to Chroma", num_indexed)

        return {"documents_indexed": num_indexed}
