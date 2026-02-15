import argparse
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain.embeddings import HuggingFaceEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Milvus Semantic Search Script")

    # Arguments for Milvus configuration
    parser.add_argument("--milvus_uri", type=str, default="http://localhost:19530", help="Milvus server URI.")
    parser.add_argument("--milvus_token", type=str, default="root:Milvus", help="Authentication token for Milvus.")
    parser.add_argument("--collection_name", required=True, help="Name of the collection to create in Milvus.")
    parser.add_argument("--dimension", type=int, default=768, help="Dimension of the vectors in the collection.")
    parser.add_argument("--metric", type=str, default="COSINE", help="Distance metric for the collection (default: COSINE).")

    # Arguments for embedding models
    parser.add_argument(
        "--dense_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Dense embedding model."
    )
    parser.add_argument("--question", type=str, required=True, help="Question to perform semantic search on.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to retrieve.")
    parser.add_argument("--partition_name", type=str, default="default_partition", help="Partition for querying Milvus.")

    args = parser.parse_args()

    # Connect to Milvus
    connections.connect("default", uri=args.milvus_uri, token=args.milvus_token)

    # Define schema and create the collection if it doesn't exist
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.dimension),
    ]
    schema = CollectionSchema(fields, "Collection schema for semantic search.")

    collection = Collection(name=args.collection_name, schema=schema, using="default")
    if args.partition_name not in [p.name for p in collection.partitions]:
        collection.create_partition(args.partition_name)

    # Initialize dense embedding model using LangChain
    text_embedder = HuggingFaceEmbeddings(model_name=args.dense_model)

    # Generate dense embedding for the question
    question_embedding = text_embedder.embed_query(args.question)

    # Perform query on Milvus
    search_results = collection.search(
        data=[question_embedding],
        anns_field="embedding",
        param={"metric_type": args.metric, "params": {"nprobe": 10}},
        limit=args.top_k,
        partition_names=[args.partition_name],
    )

    # Process and print the results
    retrieval_results = [
        {"id": res.id, "score": res.distance} for res in search_results[0]
    ]
    print(retrieval_results)


if __name__ == "__main__":
    main()
