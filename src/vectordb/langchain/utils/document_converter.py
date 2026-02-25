"""Converter utilities for Haystack to LangChain document format."""

from haystack.dataclasses import Document as HaystackDocument
from langchain_core.documents import Document as LangChainDocument


class HaystackToLangchainConverter:
    """Convert Haystack Documents to LangChain Documents.

    All vectordb database wrappers return Haystack Documents, while LangChain
    pipeline helpers (RAGHelper, RerankerHelper, MMRHelper) require LangChain
    Documents. This converter bridges the gap.
    """

    @staticmethod
    def convert(documents: list[HaystackDocument]) -> list[LangChainDocument]:
        """Convert a list of Haystack Documents to LangChain Documents.

        Args:
            documents: List of Haystack Document objects.

        Returns:
            List of LangChain Document objects with page_content and metadata.
        """
        result = []
        for doc in documents:
            meta = dict(doc.meta) if doc.meta else {}
            if doc.id:
                meta["id"] = doc.id
            if doc.score is not None:
                meta["score"] = doc.score
            result.append(
                LangChainDocument(
                    page_content=doc.content or "",
                    metadata=meta,
                )
            )
        return result

    @staticmethod
    def convert_with_embeddings(
        documents: list[HaystackDocument],
    ) -> tuple[list[LangChainDocument], list[list[float]]]:
        """Convert Haystack Documents to LangChain Documents, extracting embeddings.

        Only documents with a non-None embedding are included. Used by MMR
        pipelines that need embeddings separately for diversity scoring.

        Args:
            documents: List of Haystack Document objects.

        Returns:
            Tuple of (langchain_docs, embeddings) where embeddings[i] corresponds
            to langchain_docs[i]. Documents without embeddings are excluded.
        """
        langchain_docs = []
        embeddings = []
        for doc in documents:
            if doc.embedding is not None:
                meta = dict(doc.meta) if doc.meta else {}
                if doc.id:
                    meta["id"] = doc.id
                if doc.score is not None:
                    meta["score"] = doc.score
                langchain_docs.append(
                    LangChainDocument(
                        page_content=doc.content or "",
                        metadata=meta,
                    )
                )
                embeddings.append(doc.embedding)
        return langchain_docs, embeddings
