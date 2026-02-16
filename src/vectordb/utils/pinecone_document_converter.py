"""Pinecone Document Converter.

This module provides utilities for preparing and converting documents between
different formats for use in vector store operations with Pinecone. It includes
methods for handling Haystack and LangChain document formats and for converting
query results into Haystack documents.

Classes:
    - PineconeDocumentConverter: Contains static methods for preparing and
      transforming documents.

Methods:
    - prepare_haystack_documents_for_upsert: Prepares Haystack Document objects
      for vector store upsertion.
    - prepare_langchain_documents_for_upsert: Placeholder for preparing
      LangChain Document objects for upsertion.
    - convert_query_results_to_haystack_documents: Converts query results into
      Haystack Document objects.

Logging:
    - Uses a centralized logging utility to provide detailed logs for all
      operations.

Exceptions:
    - Raises ValueError for invalid document inputs.
"""

import logging
from typing import Any, Optional

from haystack import Document as HaystackDocument
from haystack.dataclasses import SparseEmbedding
from langchain_core.documents import Document as LangchainDocument
from langchain_qdrant.sparse_embeddings import SparseVector

from vectordb.utils.logging import LoggerFactory


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class PineconeDocumentConverter:
    """Utility class for preparing and converting documents for vector store ops."""

    @staticmethod
    def prepare_haystack_documents_for_upsert(
        documents: list[HaystackDocument],
    ) -> list[dict[str, Any]]:
        """Convert Haystack documents to Pinecone upsert format.

        This method takes a list of Haystack Document objects and prepares them
        for upsertion into a vector store. It ensures that each document has an
        ID and embeddings, and formats the data accordingly for the vector store.
        The method also handles sparse embeddings if they are present in the
        documents.

        Args:
            documents (List[HaystackDocument]): A list of Haystack Document
                objects, each containing embeddings, sparse embeddings, and
                metadata.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries formatted for vector
                store upsertion. Each dictionary includes:
                - 'id': The document ID.
                - 'values': The dense embedding of the document.
                - 'sparse_values': (Optional) Sparse embedding data, with
                  'indices' and 'values'.
                - 'metadata': Metadata, including the document content and
                  additional attributes.

        Raises:
            ValueError: If any document lacks an ID or embeddings.
        """
        upsert_data: list[dict[str, Any]] = []

        for document in documents:
            if not document.id or document.embedding is None:
                logger.error(
                    f"Document with missing ID or embedding detected: {document}"
                )
                msg = "Each document must have an ID and a dense embedding."
                raise ValueError(msg)

            prepared_doc = {
                "id": document.id,
                "values": document.embedding,
                "metadata": {"text": document.content, **document.meta},
            }

            if document.sparse_embedding is not None:
                prepared_doc["sparse_values"] = {
                    "indices": document.sparse_embedding.indices,
                    "values": document.sparse_embedding.values,
                }

            upsert_data.append(prepared_doc)
            logger.info(f"Prepared document for upsert: {document.id}")

        logger.info(
            f"Prepared {len(upsert_data)} documents for vector store upsertion."
        )
        return upsert_data

    @staticmethod
    def prepare_langchain_documents_for_upsert(
        documents: list[LangchainDocument],
        embeddings: list[list[float]],
        sparse_embeddings: Optional[list[Optional[SparseVector]]] = None,
    ) -> list[dict[str, Any]]:
        """Prepare LangChain Document objects for upserting into a vector store.

        Args:
            documents (List[LangChainDocument]): A list of LangChain Document
                objects.
            embeddings (List[List[float]]): A list of embeddings corresponding
                to the documents.
            sparse_embeddings (Optional[List[Optional[SparseVector]]]): A list
                of sparse embeddings, one per document. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries formatted for vector
                store upsertion.
        """
        if sparse_embeddings is None:
            sparse_embeddings = [None] * len(documents)

        upsert_data = [
            {
                "id": str(index),
                "values": embeddings[index],
                "metadata": {"text": document.page_content, **document.metadata},
                **(
                    {
                        "sparse_values": {
                            "indices": sparse.indices,
                            "values": sparse.values,
                        }
                    }
                    if sparse
                    else {}
                ),
            }
            for index, (document, sparse) in enumerate(
                zip(documents, sparse_embeddings)
            )
        ]

        logger.info(
            f"Prepared {len(upsert_data)} documents for vector store upsertion."
        )
        return upsert_data

    @staticmethod
    def convert_query_results_to_haystack_documents(
        query_results: dict[str, Any],
    ) -> list[HaystackDocument]:
        """Convert query results into a list of Haystack Document objects.

        This method extracts relevant information from the query results and
        constructs Haystack Document objects. It handles sparse embeddings if
        they are present in the query results.

        Args:
            query_results (Dict[str, Any]): Query results containing match
                information.

        Returns:
            List[HaystackDocument]: A list of Haystack Document objects.
        """
        haystack_docs: list[HaystackDocument] = []

        results = query_results.get("matches", [])
        if not results:
            logger.warning("Query results contain no matches.")
            return haystack_docs

        for result in results:
            metadata = result.get("metadata", {}).copy()
            text = metadata.pop("text", "")

            sparse_embedding_data = result.get("sparse_values", None)
            sparse_embedding = None
            if sparse_embedding_data:
                sparse_embedding = SparseEmbedding(
                    indices=sparse_embedding_data.get("indices", []),
                    values=sparse_embedding_data.get("values", []),
                )

            document = HaystackDocument(
                content=text,
                id=result.get("id"),
                meta=metadata,
                score=result.get("score", 0),
                embedding=result.get("values"),
                sparse_embedding=sparse_embedding,
            )

            haystack_docs.append(document)
            logger.info(f"Converted query result to HaystackDocument: {document.id}")

        logger.info(
            f"Converted {len(haystack_docs)} query results into HaystackDocument objects."
        )
        return haystack_docs

    @staticmethod
    def convert_query_results_to_langchain_documents(
        query_results: dict[str, Any],
    ) -> list[LangchainDocument]:
        """Convert query results into a list of LangChain Document objects.

        This method extracts relevant information from the query results and
        constructs LangChain Document objects.

        Args:
            query_results (Dict[str, Any]): Query results containing match
                information.

        Returns:
            List[LangChainDocument]: A list of LangChain Document objects.
        """
        langchain_docs: list[LangchainDocument] = []

        results = query_results.get("matches", [])
        if not results:
            logger.warning("Query results contain no matches.")
            return langchain_docs

        for result in results:
            metadata = result.get("metadata", {}).copy()
            text = metadata.pop("text", "")

            document = HaystackDocument(
                content=text,
                id=result.get("id"),
                meta=metadata,
            )

            langchain_docs.append(document)
            logger.info(f"Converted query result to LangChainDocument: {document.id}")

        logger.info(
            f"Converted {len(langchain_docs)} query results into LangChainDocument objects."
        )
        return langchain_docs
