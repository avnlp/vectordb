"""Semantic search script for Pinecone vector database.

This module provides functionality to perform semantic search
using Pinecone vector database with dense and sparse embeddings.
"""

import argparse

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
)

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    """Perform semantic search using Pinecone vector database.

    This function initializes Pinecone, generates dense and sparse embeddings
    for a query, and retrieves top-k similar documents.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Perform semantic search using Pinecone with dense and sparse embeddings."
    )
    parser.add_argument("--api_key", type=str, required=True, help="Pinecone API key")
    parser.add_argument(
        "--index_name", type=str, required=True, help="Pinecone index name"
    )
    parser.add_argument(
        "--namespace", type=str, required=True, help="Namespace in the Pinecone index"
    )
    parser.add_argument(
        "--dense_model", type=str, required=True, help="Dense embedding model name"
    )
    parser.add_argument(
        "--sparse_model", type=str, required=True, help="Sparse embedding model name"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Query or question for semantic search",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve"
    )
    args = parser.parse_args()

    # Initialize Pinecone vector database
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.api_key, index_name=args.index_name
    )

    # Initialize dense embedder
    text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    text_embedder.warm_up()

    # Initialize sparse embedder
    sparse_text_embedder = FastembedSparseTextEmbedder(model=args.sparse_model)
    sparse_text_embedder.warm_up()

    # Generate embeddings for the query
    dense_question_embedding = text_embedder.run(text=args.question)["embedding"]
    sparse_question_embedding = sparse_text_embedder.run(text=args.question)[
        "sparse_embedding"
    ].to_dict()

    # Perform query on Pinecone vector database
    query_response = pinecone_vector_db.query(
        vector=dense_question_embedding,
        sparse_vector=sparse_question_embedding,
        top_k=args.top_k,
        include_metadata=True,
        namespace=args.namespace,
    )

    # Convert results and print
    retrieval_results = (
        PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )
    print(retrieval_results)


if __name__ == "__main__":
    main()
