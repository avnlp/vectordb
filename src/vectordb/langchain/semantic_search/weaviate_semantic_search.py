import argparse

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from vectordb import WeaviateDocumentConverter, WeaviateVectorDB


# Argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a query using Weaviate VectorDB and HuggingFace embeddings."
    )
    parser.add_argument(
        "--cluster_url", type=str, required=True, help="Weaviate cluster URL."
    )
    parser.add_argument(
        "--api_key", type=str, required=True, help="API key for Weaviate access."
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="test_collection_dense1",
        help="Collection name in Weaviate.",
    )
    parser.add_argument(
        "--dense_embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model name.",
    )
    parser.add_argument(
        "--sparse_embedding_model",
        type=str,
        default="prithivida/Splade_PP_en_v1",
        help="Sparse embedding model name.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The question for which to retrieve answers.",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="test_namespace",
        help="Namespace for Pinecone query.",
    )
    return parser.parse_args()


# Main function
def main():
    # Parse arguments
    args = parse_args()

    # Initialize Weaviate vector database
    weaviate_vector_db = WeaviateVectorDB(
        cluster_url=args.cluster_url,
        api_key=args.api_key,
        collection_name=args.collection_name,
    )

    # Initialize text embedder for dense and sparse embeddings
    text_embedder = HuggingFaceEmbeddings(model_name=args.dense_embedding_model)
    sparse_text_embedder = FastEmbedSparse(model=args.sparse_embedding_model)

    # The question for querying
    question = args.question

    # Generate embeddings for the question
    dense_question_embedding = text_embedder.embed_query(question)
    sparse_question_embedding = sparse_text_embedder.embed_query(question)

    # Query the Weaviate database
    query_response = weaviate_vector_db.query(
        vector=dense_question_embedding,
        sparse_vector={
            "indices": sparse_question_embedding.indices,
            "values": sparse_question_embedding.values,
        },
        top_k=args.top_k,
        include_metadata=True,
        namespace=args.namespace,
    )

    # Convert query results to Haystack documents
    retrieval_results = (
        WeaviateDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )

    # Print the retrieved results
    print(retrieval_results)


if __name__ == "__main__":
    main()
