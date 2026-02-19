"""Pinecone document converter for Haystack and LangChain integration.

This module provides bidirectional conversion between Haystack/LangChain document
formats and Pinecone's native upsert/query formats. It handles dense embeddings,
sparse embeddings for hybrid search, and metadata serialization.

Key Responsibilities:
    - Convert Haystack Documents to Pinecone upsert format (id, values, metadata,
      sparse_values)
    - Convert LangChain Documents to Pinecone upsert format
    - Transform Pinecone query results (matches) back to Haystack/LangChain Documents
    - Handle hybrid search with optional sparse_values for BM25-style retrieval

Pinecone-Specific Behavior:
    - Uses flat dictionary format with 'id', 'values', 'metadata', 'sparse_values'
    - Document content stored in metadata['text'] field
    - Sparse values use {"indices": [...], "values": [...]} format
    - Score is returned directly from match results (higher is better)
    - Query results nested under 'matches' key in response

Usage:
    >>> from vectordb.utils import PineconeDocumentConverter
    >>> converter = PineconeDocumentConverter()
    >>> upsert_data = converter.prepare_haystack_documents_for_upsert(documents)
    >>> # upsert_data is list of dicts with 'id', 'values', 'metadata' keys
"""

import logging
from typing import Any, Optional

from haystack import Document as HaystackDocument
from haystack.dataclasses import SparseEmbedding
from langchain_core.documents import Document as LangchainDocument
from langchain_qdrant.sparse_embeddings import SparseVector

from vectordb.utils.ids import get_doc_id, set_doc_id
from vectordb.utils.logging import LoggerFactory
from vectordb.utils.sparse import get_doc_sparse_embedding


logger_factory = LoggerFactory(logger_name=__name__, log_level=logging.INFO)
logger = logger_factory.get_logger()


class PineconeDocumentConverter:
    """Bidirectional converter between Haystack/LangChain and Pinecone formats.

    This converter handles the format transformation required for storing documents
    in Pinecone and retrieving them back. Pinecone uses a flat dictionary format
    with specific keys ('id', 'values', 'metadata', 'sparse_values') that differ
    from Haystack's Document class structure.

    Key Transformations:
        - Haystack Document.id -> Pinecone 'id' field
        - Haystack Document.embedding -> Pinecone 'values' field
        - Haystack Document.content -> Pinecone metadata['text'] field
        - Haystack Document.meta -> Pinecone metadata (merged with text)
        - Haystack Document.sparse_embedding -> Pinecone 'sparse_values' field

    Pinecone-Specific Notes:
        - Document content is stored in metadata['text'] (not at top level)
        - Sparse values use {"indices": [...], "values": [...]} format
        - Query results are nested under 'matches' key in response
        - Scores are returned directly (higher is better, no normalization needed)

    The converter is stateless and all methods are static, allowing direct class
    method calls without instantiation.
    """

    @staticmethod
    def prepare_haystack_documents_for_upsert(
        documents: list[HaystackDocument],
    ) -> list[dict[str, Any]]:
        """Convert Haystack documents to Pinecone upsert format.

        This method takes a list of Haystack Document objects and prepares them
        for upsertion into a vector store. It ensures that each document has an
        ID and embeddings, and formats the data accordingly for the vector store.
        The method also handles sparse embeddings if present.

        Args:
            documents (List[HaystackDocument]): A list of Haystack Document
                objects, each containing embeddings, sparse embeddings,
                and metadata.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries for vector store
                upsertion. Includes 'id', 'values', 'sparse_values' (optional),
                and 'metadata'.

        Raises:
            ValueError: If any document lacks an ID or embeddings.
        """
        upsert_data: list[dict[str, Any]] = []

        for document in documents:
            if document.embedding is None:
                logger.error(f"Document with missing embedding detected: {document}")
                msg = "Each document must have a dense embedding."
                raise ValueError(msg)

            # Use utility to get ID or generate one
            doc_id = get_doc_id(document)

            prepared_doc = {
                "id": doc_id,
                "values": document.embedding,
                "metadata": {"text": document.content, **(document.meta or {})},
            }

            # Use utility to get sparse embedding
            sparse_emb = get_doc_sparse_embedding(document)
            if sparse_emb is not None:
                prepared_doc["sparse_values"] = {
                    "indices": list(sparse_emb.indices),
                    "values": list(sparse_emb.values),
                }

            upsert_data.append(prepared_doc)
            logger.info(f"Prepared document for upsert: {doc_id}")

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
            documents (List[LangChainDocument]): A list of LangChain Documents.
            embeddings (List[List[float]]): Embeddings corresponding to docs.
            sparse_embeddings (Optional[List[Optional[SparseVector]]]): Sparse
                embeddings, one per document. Defaults to None.

        Returns:
            List[Dict[str, Any]]: Dictionaries formatted for vector store upsert.
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
        include_embeddings: bool = False,
    ) -> list[HaystackDocument]:
        """Convert query results into a list of Haystack Document objects.

        This method extracts relevant information from the query results and
        constructs Haystack Document objects. It handles sparse embeddings if
        they are present in the query results.

        Args:
            query_results (Dict[str, Any]): Query results with match information.
            include_embeddings (bool): Whether to include vectors in documents.

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
                sparse_embedding=sparse_embedding,
            )

            set_doc_id(document, str(result.get("id")))

            if include_embeddings and "values" in result:
                document.embedding = result.get("values")

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
        """Convert Pinecone query results to LangChain Documents.

        Pinecone returns query results under the 'matches' key, where each match
        contains 'id', 'values', 'metadata', and 'score'. This method extracts
        the match information and constructs LangChain Document objects.

        Args:
            query_results: Pinecone query result dictionary with 'matches' key.
                Each match contains 'id', 'values' (embedding), 'metadata', and
                'score' fields.

        Returns:
            List of LangChain Documents with page_content and metadata. The
            page_content is extracted from metadata['text'], which is where
            the document content is stored during upsert. The document ID is
            placed in metadata['id'] for LangChain compatibility.

        Note:
            Unlike the Haystack version, this method does not include embeddings,
            sparse embeddings, or scores in the output, as LangChain Documents do
            not have these fields by default. The ID is stored in metadata instead
            of a top-level ID field.
        """
        langchain_docs: list[LangchainDocument] = []

        results = query_results.get("matches", [])
        if not results:
            logger.warning("Query results contain no matches.")
            return langchain_docs

        for result in results:
            metadata = result.get("metadata", {}).copy()
            text = metadata.pop("text", "")

            document = LangchainDocument(
                page_content=text,
                metadata={"id": result.get("id"), **metadata},
            )

            langchain_docs.append(document)
            logger.info(
                f"Converted query result to LangChainDocument: {document.metadata.get('id')}"
            )

        logger.info(
            f"Converted {len(langchain_docs)} query results into LangChainDocument objects."
        )
        return langchain_docs
