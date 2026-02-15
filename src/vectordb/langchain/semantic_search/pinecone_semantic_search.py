import argparse

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Query a Pinecone index using dense and sparse embeddings."
    )

    # Add arguments
    parser.add_argument("--question", required=True, help="The question to query.")
    parser.add_argument(
        "--dense_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model.",
    )
    parser.add_argument(
        "--sparse_model",
        default="prithivida/Splade_PP_en_v1",
        help="Sparse embedding model.",
    )
    parser.add_argument("--pinecone_api_key", required=True, help="Pinecone API key.")
    parser.add_argument(
        "--index_name", default="test-index-hybrid", help="Name of the Pinecone index."
    )
    parser.add_argument(
        "--namespace", default="test_namespace", help="Namespace for Pinecone queries."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve."
    )

    args = parser.parse_args()

    # Initialize Pinecone vector database
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
    )

    # Initialize embeddings
    text_embedder = HuggingFaceEmbeddings(model_name=args.dense_model)
    sparse_text_embedder = FastEmbedSparse(model=args.sparse_model)

    # Embed the query
    dense_question_embedding = text_embedder.embed_query(args.question)
    sparse_question_embedding = sparse_text_embedder.embed_query(args.question)

    # Query Pinecone
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

    # Convert query results
    retrieval_results = (
        PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )

    # Output results
    print("Retrieved Documents:")
    for idx, doc in enumerate(retrieval_results, start=1):
        print(f"{idx}. {doc.content}")


if __name__ == "__main__":
    main()
