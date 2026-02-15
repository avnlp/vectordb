"""Chroma semantic search example with LangChain integration.

This module demonstrates how to perform semantic search using Chroma VectorDB
with dense and sparse embeddings.
"""

import argparse

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from vectordb import ChromaVectorDB, PineconeDocumentConverter, PineconeVectorDB


def main():
    """Run the Chroma semantic search query.

    This function:
    - Parses command line arguments
    - Initializes Chroma and Pinecone VectorDB
    - Generates dense and sparse query embeddings
    - Queries Pinecone and retrieves results
    - Converts and prints the results
    """
    parser = argparse.ArgumentParser(
        description="Run Dense and Sparse Query on Pinecone or Chroma VectorDB."
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

    # Embedding model arguments
    parser.add_argument(
        "--dense_embedding_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model.",
    )
    parser.add_argument(
        "--sparse_embedding_model",
        default="prithivida/Splade_PP_en_v1",
        help="Sparse embedding model.",
    )

    # Query parameters
    parser.add_argument("--question", required=True, help="The question to be queried.")
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results to retrieve from the vector database.",
    )
    parser.add_argument(
        "--namespace", default="test_namespace", help="Namespace for Pinecone."
    )

    # Pinecone API arguments
    parser.add_argument("--pinecone_api_key", required=True, help="Pinecone API key.")
    parser.add_argument("--pinecone_index", required=True, help="Pinecone index name.")

    args = parser.parse_args()

    # Initialize Chroma VectorDB
    chroma_vector_db = ChromaVectorDB(persistent=True, path=args.chroma_path)
    chroma_vector_db.create_collection(name=args.chroma_collection)

    # Initialize the embedding models
    text_embedder = HuggingFaceEmbeddings(model_name=args.dense_embedding_model)
    sparse_text_embedder = FastEmbedSparse(model=args.sparse_embedding_model)

    # Get the dense and sparse embeddings for the query
    dense_question_embedding = text_embedder.embed_query(args.question)
    sparse_question_embedding = sparse_text_embedder.embed_query(args.question)

    # Initialize Pinecone VectorDB and query
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.pinecone_api_key, index_name=args.pinecone_index
    )

    query_response = pinecone_vector_db.query(
        vector=dense_question_embedding,
        sparse_vector={
            "indices": sparse_question_embedding.indices,
            "values": sparse_question_embedding.values,
        },
        top_k=args.top_k,
        include_metadata=True,
        namespace=args.namespace,
    )

    retrieval_results = (
        PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )
    print(retrieval_results)


if __name__ == "__main__":
    main()
