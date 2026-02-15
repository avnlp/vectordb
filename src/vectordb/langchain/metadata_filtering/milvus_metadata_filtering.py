import argparse
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema, HybridSearchRequest, Filter
from vectordb import MilvusDocumentConverter


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Query Milvus VectorDB using HuggingFace embeddings.")
    parser.add_argument(
        "--milvus_host",
        type=str,
        required=True,
        help="Milvus server host.",
    )
    parser.add_argument(
        "--milvus_port",
        type=str,
        default="19530",
        help="Milvus server port (default is 19530).",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="Collection name in Milvus.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="HuggingFace embedding model name (default is 'sentence-transformers/all-mpnet-base-v2').",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to query the database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of top results to retrieve (default is 10).",
    )
    parser.add_argument(
        "--hybrid",
        type=bool,
        default=True,
        help="Use hybrid search (default is True).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid search alpha value (default is 0.5).",
    )
    args = parser.parse_args()

    # Initialize Milvus client
    client = MilvusClient(host=args.milvus_host, port=args.milvus_port)

    # Check if the collection exists
    if not client.has_collection(args.collection_name):
        raise ValueError(f"Collection '{args.collection_name}' does not exist in Milvus.")

    # Initialize text embedder
    text_embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)

    # Generate embedding for the query
    dense_question_embedding = text_embedder.embed_query(args.question)

    # Build the hybrid search request
    dense_search_request = {
        "anns_field": "dense",
        "data": [dense_question_embedding],
        "param": {"metric_type": "IP", "params": {"nprobe": 10}},
        "limit": args.limit,
    }

    # If hybrid search is enabled, include a sparse vector field (if applicable)
    if args.hybrid:
        hybrid_search_request = HybridSearchRequest(
            dense_search_request=dense_search_request,
            alpha=args.alpha
        )
        search_response = client.hybrid_search(
            collection_name=args.collection_name, 
            hybrid_search_request=hybrid_search_request
        )
    else:
        search_response = client.search(
            collection_name=args.collection_name,
            anns_field=dense_search_request["anns_field"],
            data=dense_search_request["data"],
            param=dense_search_request["param"],
            limit=dense_search_request["limit"]
        )

    # Convert and print the results
    retrieval_results = MilvusDocumentConverter.convert_query_results_to_haystack_documents(search_response)
    print(retrieval_results)


if __name__ == "__main__":
    main()
