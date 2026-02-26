"""Qdrant query enhancement search pipeline (LangChain)."""

from typing import Any

from langchain_core.documents import Document

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.langchain.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class QdrantQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Qdrant query enhancement search pipeline (LangChain).

    Enhances query with multiple perspectives, performs parallel searches,
    and fuses results using RRF.
    """

    @property
    def _db_key(self) -> str:
        return "qdrant"

    def _initialize_db(self, db_config: dict[str, Any]) -> QdrantVectorDB:
        self.collection_name = db_config.get("collection_name")
        return QdrantVectorDB(
            url=db_config.get("url"),
            api_key=db_config.get("api_key"),
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
            collection_name=self.collection_name,
        )
