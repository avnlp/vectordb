"""Weaviate query enhancement search pipeline (LangChain)."""

from typing import Any

from langchain_core.documents import Document

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class WeaviateQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Weaviate query enhancement search pipeline (LangChain).

    Enhances query with multiple perspectives, performs parallel searches,
    and fuses results using RRF.
    """

    @property
    def _db_key(self) -> str:
        return "weaviate"

    def _initialize_db(self, db_config: dict[str, Any]) -> WeaviateVectorDB:
        self.collection_name = db_config.get("collection_name")
        return WeaviateVectorDB(
            url=db_config["url"],
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
