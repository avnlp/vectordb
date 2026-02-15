"""Semantic search script for Weaviate vector database.

This module provides functionality to perform semantic search
using Weaviate vector database with dense embeddings.
"""

import argparse

from haystack.components.embedders import SentenceTransformersTextEmbedder
from pinecone import ServerlessSpec

from vectordb import WeaviateDocumentConverter, WeaviateVectorDB


def main():
    """Perform semantic search using Weaviate vector database.

    This function initializes Weaviate, generates embeddings for a query,
    and retrieves top-k similar documents.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Weaviate Semantic Search Script")

    # Arguments for Weaviate configuration
    parser.add_argument(
        "--api_key", type=str, required=True, help="API key for Weaviate access."
    )
    parser.add_argument(
        "--index_name", required=True, help="Name of the index to create in Weaviate."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Dimension of the vectors in the index.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric for the index (default: cosine).",
    )
    parser.add_argument(
        "--cloud",
        type=str,
        default="aws",
        help="Cloud provider for Weaviate serverless setup (default: aws).",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Region for Weaviate serverless setup (default: us-east-1).",
    )

    # Arguments for embedding models
    parser.add_argument(
        "--dense_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to perform semantic search on.",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="test_namespace",
        help="Namespace for querying Weaviate.",
    )

    args = parser.parse_args()

    # Initialize Weaviate vector DB
    weaviate_vector_db = WeaviateVectorDB(api_key=args.api_key)
    weaviate_vector_db.create_index(
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        spec=ServerlessSpec(cloud=args.cloud, region=args.region),
    )

    # Initialize dense embedding model
    text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    text_embedder.warm_up()

    # Generate dense embedding for the question
    question_embedding = text_embedder.run(text=args.question)["embedding"]

    # Perform query on Weaviate
    query_response = weaviate_vector_db.query(
        vector=question_embedding,
        top_k=args.top_k,
        include_metadata=True,
        namespace=args.namespace,
    )

    # Convert query results to Haystack documents and print
    retrieval_results = (
        WeaviateDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )
    print(retrieval_results)


if __name__ == "__main__":
    main()
