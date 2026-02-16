"""Milvus reranking example with LangChain integration.

This module demonstrates how to perform hybrid search with reranking using Milvus
with various reranking strategies (Weighted and RRF).
"""

import argparse

from dataloaders import TriviaQADataloader
from langchain.schema import Document
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker, WeightedRanker


def main():
    """Perform hybrid search with reranking in Milvus."""
    parser = argparse.ArgumentParser(
        description="Hybrid Search in Milvus with Argparse"
    )

    # Milvus Configuration
    parser.add_argument(
        "--uri", type=str, default="http://localhost:19530", help="Milvus server URI"
    )
    parser.add_argument(
        "--token",
        type=str,
        default="root:Milvus",
        help="Authentication token for Milvus",
    )
    parser.add_argument(
        "--collection_name", type=str, required=True, help="Name of the collection"
    )

    # Schema Configuration
    parser.add_argument("--dim", type=int, default=5, help="Dimension of dense vectors")
    parser.add_argument(
        "--index_type",
        type=str,
        default="IVF_FLAT",
        help="Type of index for dense vectors",
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        default="IP",
        help="Metric type for vector similarity",
    )

    # Query Parameters
    parser.add_argument(
        "--nprobe", type=int, default=10, help="Number of probe cells for ANN search"
    )
    parser.add_argument(
        "--limit", type=int, default=2, help="Number of top results to retrieve"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the TriviaQA dataset"
    )
    parser.add_argument(
        "--split", type=str, default="test[:5]", help="Split of the dataset to use"
    )

    # Reranker Configuration
    parser.add_argument(
        "--reranker",
        type=str,
        choices=["weighted", "rrf"],
        default="weighted",
        help="Reranker strategy",
    )
    parser.add_argument(
        "--weighted_params",
        type=float,
        nargs=2,
        default=[0.8, 0.3],
        help="Parameters for WeightedRanker (weights for dense and sparse)",
    )
    parser.add_argument(
        "--rrf_param", type=int, default=100, help="Parameter for RRFRanker"
    )

    args = parser.parse_args()

    # Connect to Milvus
    client = MilvusClient(uri=args.uri, token=args.token)

    # Create or load collection schema
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=args.dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense",
        index_name="dense_index",
        index_type=args.index_type,
        metric_type=args.metric_type,
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="sparse",
        index_name="sparse_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": 0.2},
    )

    client.create_collection(
        collection_name=args.collection_name, schema=schema, index_params=index_params
    )

    # Load data using the dataloader
    dataloader = TriviaQADataloader(
        answer_summary_generator=None,  # Replace with actual generator if needed
        dataset_name=args.dataset_name,
        split=args.split,
    )
    dataloader.load_data()
    data = dataloader.get_data()

    # Insert data into Milvus
    dense_vectors = []
    sparse_vectors = []
    texts = []
    for item in data:
        dense_vectors.append(item["dense_vector"])
        sparse_vectors.append(item["sparse_vector"])
        texts.append(item["text"])

    client.insert(
        collection_name=args.collection_name,
        data={
            "dense": dense_vectors,
            "sparse": sparse_vectors,
            "text": texts,
        },
    )
    client.flush(args.collection_name)

    # Prepare query vectors (example: using the first entry from the data loader)
    dense_query_vector = dense_vectors[0]
    sparse_query_vector = sparse_vectors[0]

    # Create search requests
    search_param_1 = {
        "data": [dense_query_vector],
        "anns_field": "dense",
        "param": {"metric_type": args.metric_type, "params": {"nprobe": args.nprobe}},
        "limit": args.limit,
    }
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data": [sparse_query_vector],
        "anns_field": "sparse",
        "param": {"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},
        "limit": args.limit,
    }
    request_2 = AnnSearchRequest(**search_param_2)

    reqs = [request_1, request_2]

    # Configure reranker
    if args.reranker == "weighted":
        ranker = WeightedRanker(*args.weighted_params)
    elif args.reranker == "rrf":
        ranker = RRFRanker(args.rrf_param)

    # Perform hybrid search
    res = client.hybrid_search(
        collection_name=args.collection_name, reqs=reqs, ranker=ranker, limit=args.limit
    )

    # Convert results to LangChain documents
    langchain_docs = []
    for hits in res:
        for hit in hits:
            langchain_docs.append(
                Document(
                    page_content=hit.entity.get("text"), metadata={"score": hit.score}
                )
            )

    # Display LangChain documents
    for doc in langchain_docs:
        print(f"Content: {doc.page_content}\nMetadata: {doc.metadata}")


if __name__ == "__main__":
    main()
