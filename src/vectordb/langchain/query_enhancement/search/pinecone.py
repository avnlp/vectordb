"""Pinecone query enhancement search pipeline (LangChain)."""

from typing import Any

from langchain_core.documents import Document

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class PineconeQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Pinecone query enhancement search pipeline (LangChain).

    Enhances query with multiple perspectives, performs parallel searches,
    and fuses results using RRF.
    """

    @property
    def _db_key(self) -> str:
        return "pinecone"

    def _initialize_db(self, db_config: dict[str, Any]) -> PineconeVectorDB:
        self.namespace = db_config.get("namespace", "")
        return PineconeVectorDB(
            api_key=db_config["api_key"],
            index_name=db_config.get("index_name"),
        )

    def _perform_search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None,
    ) -> list[Document]:
        return self.db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            namespace=self.namespace,
        )
