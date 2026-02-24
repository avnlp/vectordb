"""Pinecone indexing pipeline for query enhancement feature."""

from pathlib import Path
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.query_enhancement.indexing.base import (
    BaseQueryEnhancementIndexingPipeline,
)


class PineconeQueryEnhancementIndexingPipeline(BaseQueryEnhancementIndexingPipeline):
    """Index documents into Pinecone for query enhancement retrieval.

    Loads documents from configured dataloader, embeds them,
    and upserts to Pinecone index.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "pinecone_query_enhancement_indexing")

    def _init_db(self) -> PineconeVectorDB:
        """Initialize Pinecone VectorDB from config."""
        pinecone_config = self.config.get("pinecone", {})
        return PineconeVectorDB(
            api_key=pinecone_config.get("api_key"),
            index_name=pinecone_config.get("index_name"),
            config=self.config,
        )

    def run(self) -> dict[str, Any]:
        """Execute the indexing pipeline.

        Returns:
            Dictionary with indexing statistics.
        """
        self.logger.info("Starting document indexing")

        documents = self._dataset.to_haystack()
        self.logger.info(f"Loaded {len(documents)} documents")

        # Embed documents
        embedded = self.embedder.run(documents=documents)
        docs_with_embeddings = embedded["documents"]
        self.logger.info(f"Embedded {len(docs_with_embeddings)} documents")

        if docs_with_embeddings and docs_with_embeddings[0].embedding:
            dimension = len(docs_with_embeddings[0].embedding)
            self.db.create_index(dimension=dimension)

        # Upsert to Pinecone with namespace support
        namespace = self.config.get("pinecone", {}).get("namespace", "default")
        count = self.db.upsert(docs_with_embeddings, namespace=namespace)

        self.logger.info(f"Indexed {count} documents to namespace '{namespace}'")
        return {"documents_indexed": count, "namespace": namespace}
