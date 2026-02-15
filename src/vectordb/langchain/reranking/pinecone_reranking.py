import argparse

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Semantic search and reranking pipeline using Pinecone."
    )

    # Query and embedding arguments
    parser.add_argument("--question", required=True, help="The question to query.")
    parser.add_argument(
        "--dense_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model name.",
    )
    parser.add_argument(
        "--sparse_model",
        default="prithivida/Splade_PP_en_v1",
        help="Sparse embedding model name.",
    )

    # Pinecone arguments
    parser.add_argument("--pinecone_api_key", required=True, help="Pinecone API key.")
    parser.add_argument(
        "--index_name", default="test-index-hybrid", help="Pinecone index name."
    )
    parser.add_argument(
        "--namespace", default="test_namespace", help="Namespace for Pinecone queries."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to retrieve from Pinecone.",
    )
    parser.add_argument(
        "--rerank_model", default="pinecone-rerank-v0", help="Reranking model to use."
    )
    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=4,
        help="Number of results to return after reranking.",
    )

    args = parser.parse_args()

    # Initialize Pinecone
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
    )

    # Initialize embeddings
    text_embedder = HuggingFaceEmbeddings(model_name=args.dense_model)
    sparse_text_embedder = FastEmbedSparse(model=args.sparse_model)

    # Create query embeddings
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

    # Convert results
    retrieval_results = (
        PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )

    # Perform reranking
    reranked_result = pinecone_vector_db.client.inference.rerank(
        model=args.rerank_model,
        query=args.question,
        documents=[doc.content for doc in retrieval_results],
        top_n=args.rerank_top_n,
        return_documents=True,
        parameters={"truncate": "END"},
    )

    # Print results
    print("Reranked Results:")
    for idx, doc in enumerate(reranked_result["documents"], start=1):
        print(f"{idx}. {doc}")


if __name__ == "__main__":
    main()
