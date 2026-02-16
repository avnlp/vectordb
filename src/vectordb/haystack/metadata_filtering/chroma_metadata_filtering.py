"""Metadata filtering script for Chroma vector database.

This module provides functionality to query Chroma vector database
with metadata filtering using Haystack components.
"""

import argparse
from ast import literal_eval

from haystack.components.embedders import SentenceTransformersTextEmbedder

from vectordb import ChromaDocumentConverter, ChromaVectorDB


def main():
    """Perform Metadata Filtering using Chroma."""
    parser = argparse.ArgumentParser(
        description="Query Chroma vector database with a dense embedding."
    )

    # Chroma VectorDB arguments
    parser.add_argument(
        "--chroma_path",
        default="./chroma_database_files",
        help="Path for Chroma database files.",
    )
    parser.add_argument(
        "--chroma_collection",
        default="test_collection_dense1",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--tracing_project_name",
        default="chroma",
        help="Name of the Weave project for tracing.",
    )
    parser.add_argument(
        "--weave_params",
        type=str,
        help="JSON string of parameters for initializing Weave.",
    )

    # Generator parameters
    parser.add_argument(
        "--generator_model", type=str, help="Model name for the dataloader's generator."
    )
    parser.add_argument(
        "--generator_api_key", help="API key for the dataloader generator."
    )
    parser.add_argument(
        "--generator_llm_params",
        type=str,
        help="JSON string of parameters for the generator LLM.",
    )

    # Embedder parameters
    parser.add_argument(
        "--embedding_model",
        type=str,
        help="Model to use for generating document embeddings.",
    )
    parser.add_argument(
        "--embedding_model_params",
        type=str,
        help="JSON string of parameters for the embedding model.",
    )

    # Query arguments
    parser.add_argument(
        "--question", required=True, help="Question to query the Chroma database."
    )
    parser.add_argument(
        "--query_filter", type=str, help="JSON string for query filter."
    )
    parser.add_argument(
        "--n_results", type=int, default=10, help="Number of results to retrieve."
    )

    args = parser.parse_args()

    # Initialize Chroma VectorDB
    weave_params = literal_eval(args.weave_params) if args.weave_params else {}
    chroma_vector_db = ChromaVectorDB(
        persistent=True,
        path=args.chroma_path,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )
    chroma_vector_db = ChromaVectorDB(path=args.chroma_path)
    chroma_vector_db.create_collection(name=args.chroma_collection)

    # Load the embedding model
    text_embedder = SentenceTransformersTextEmbedder(model=args.embedding_model)
    text_embedder.warm_up()

    # Generate dense embedding for the query
    dense_question_embedding = text_embedder.run(text=args.question)["embedding"]

    # Query the Chroma VectorDB
    query_filter = eval(args.query_filter)
    query_response = chroma_vector_db.query(
        query_embedding=dense_question_embedding,
        n_results=args.n_results,
        where_document=query_filter,
    )
    print(query_response)

    # Convert query results to Haystack documents
    retrieval_results = (
        ChromaDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )
    print(retrieval_results)


if __name__ == "__main__":
    main()
