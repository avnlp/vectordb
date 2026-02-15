import argparse

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
)

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    """Perform Metadata Filtering using Pinecone.

    This script:
    - Initializes dense and sparse embedders.
    - Generates embeddings for a query.
    - Queries a Pinecone vector database using the embeddings.
    - Prints the retrieval results.
    """
    parser = argparse.ArgumentParser(
        description="Hybrid embedding and retrieval using Pinecone"
    )
    parser.add_argument("--api_key", type=str, required=True, help="Pinecone API key")
    parser.add_argument(
        "--index_name", type=str, required=True, help="Name of the Pinecone index"
    )
    parser.add_argument(
        "--dense_model", type=str, required=True, help="Model name for dense embeddings"
    )
    parser.add_argument(
        "--sparse_model",
        type=str,
        required=True,
        help="Model name for sparse embeddings",
    )
    parser.add_argument(
        "--question", type=str, required=True, help="The query/question text"
    )
    parser.add_argument(
        "--namespace", type=str, required=True, help="Namespace for Pinecone index"
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve"
    )
    parser.add_argument(
        "--filter", type=str, help="Filter for Pinecone query in JSON format"
    )
    parser.add_argument(
        "--tracing_project_name", type=str, help="Name of the tracing project."
    )
    parser.add_argument(
        "--weave_params",
        type=str,
        help="JSON string of parameters for configuring Weave.",
    )
    args = parser.parse_args()

    # Initialize Pinecone vector database
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.api_key,
        index_name=args.index_name,
    )

    # Initialize dense embedder
    text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    text_embedder.warm_up()

    # Initialize sparse embedder
    sparse_embedder = FastembedSparseTextEmbedder(model=args.sparse_model)
    sparse_embedder.warm_up()

    # Generate embeddings for the query
    dense_question_embedding = text_embedder.run(text=args.question)["embedding"]
    sparse_question_embedding = sparse_embedder.run(text=args.question)[
        "sparse_embedding"
    ].to_dict()

    # Query Pinecone vector database
    query_response = pinecone_vector_db.query(
        vector=dense_question_embedding,
        sparse_vector=sparse_question_embedding,
        top_k=args.top_k,
        include_metadata=True,
        namespace=args.namespace,
        filter=args.filter,
    )

    # Convert and print results
    retrieval_results = (
        PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )
    print(retrieval_results)


if __name__ == "__main__":
    main()
