"""Weaviate BM25 indexing pipeline for keyword/BM25-style search.

Note: Weaviate does not support external sparse vectors like SPLADE.
Instead, it computes BM25 internally from the stored text at query time.
"""

from pathlib import Path
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.utils import ConfigLoader
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


class WeaviateBM25IndexingPipeline:
    """Weaviate BM25 indexing pipeline.

    Weaviate computes BM25 internally from stored text at query time,
    so no external sparse embeddings are needed.
    """

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["weaviate"]

        self.db = WeaviateVectorDB(
            url=db_config.get("url"),
            api_key=db_config.get("api_key"),
            index_name=db_config.get("index_name"),
        )

        self.batch_size = self.config.get("indexing", {}).get("batch_size", 100)

        logger.info(
            f"Initialized WeaviateBM25IndexingPipeline with index: {db_config.get('index_name')}"
        )

    def create_collection(self) -> None:
        """Create collection with BM25 support."""
        self.db.create_collection()
        logger.info("Created Weaviate collection with BM25 support")

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

        # 2. Prepare for Weaviate upsert (no sparse embedding needed)
        # Weaviate computes BM25 internally from text at query time
        for i, doc in enumerate(documents):
            if not doc.id:
                doc.id = f"doc_{i}"

        # 3. Upsert to Weaviate
        self.db.upsert(
            documents=documents,
            batch_size=self.batch_size,
        )

        logger.info(f"Indexed {len(documents)} documents to Weaviate")
        return {"documents_indexed": len(documents)}
