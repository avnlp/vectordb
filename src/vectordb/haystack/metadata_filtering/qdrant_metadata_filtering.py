import argparse
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, ScoredPoint, Filter


def main():
    """Perform Metadata Filtering using Qdrant.

    This script:
    - Initializes dense and sparse embedders.
    - Generates embeddings for a query.
    - Queries a Qdrant vector database using the embeddings.
    - Prints the retrieval results.
    """
    parser = argparse.ArgumentParser(description="Hybrid embedding and retrieval using Qdrant")
    parser.add_argument("--qdrant_url", type=str, required=True, help="Qdrant URL")
    parser.add_argument("--collection_name", type=str, required=True, help="Name of the Qdrant collection")
    parser.add_argument("--dense_model", type=str, required=True, help="Model name for dense embeddings")
    parser.add_argument("--sparse_model", type=str, required=True, help="Model name for sparse embeddings")
    parser.add_argument("--question", type=str, required=True, help="The query/question text")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to retrieve")
    parser.add_argument("--filter", type=str, help="Filter for Qdrant query in JSON format")
    args = parser.parse_args()

    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=args.qdrant_url)

    # Initialize dense embedder
    text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    text_embedder.warm_up()

    # Initialize sparse embedder
    sparse_embedder = FastembedSparseTextEmbedder(model=args.sparse_model)
    sparse_embedder.warm_up()

    # Generate embeddings for the query
    dense_question_embedding = text_embedder.run(text=args.question)["embedding"]
    sparse_question_embedding = sparse_embedder.run(text=args.question)["sparse_embedding"].to_dict()

    # Prepare Qdrant filter if provided
    filter = None
    if args.filter:
        filter = Filter.from_dict(eval(args.filter))  

    # Query Qdrant vector database for both dense and sparse embeddings
    query_response = qdrant_client.search(
        collection_name=args.collection_name,
        query_vector=dense_question_embedding,
        query_sparse_vector=sparse_question_embedding,
        limit=args.top_k,
        filter=filter,
    )

    # Process and print results
    retrieval_results = []
    for hit in query_response:
        # Convert Qdrant search result to Haystack document format
        retrieval_results.append({
            "id": hit.id,
            "score": hit.score,
            "metadata": hit.payload, 
        })

    # Output the results
    print(retrieval_results)


if __name__ == "__main__":
    main()
