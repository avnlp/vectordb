"""Milvus query enhancement search pipeline (LangChain)."""

from typing import Any

from langchain_core.documents import Document

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class MilvusQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Milvus query enhancement search pipeline (LangChain).

    Enhances query with multiple perspectives, performs parallel searches,
    and fuses results using RRF.
    """

    @property
    def _db_key(self) -> str:
        return "milvus"

    def _initialize_db(self, db_config: dict[str, Any]) -> MilvusVectorDB:
        self.collection_name = db_config.get("collection_name")
        return MilvusVectorDB(
            host=db_config.get("host"),
            port=db_config.get("port"),
            db_name=db_config.get("db_name"),
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
