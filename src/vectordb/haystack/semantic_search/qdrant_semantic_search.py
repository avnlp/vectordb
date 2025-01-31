import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from haystack.components.embedders import SentenceTransformersTextEmbedder

from vectordb import QdrantDocumentConverter, QdrantVectorDB


def main():
    parser = argparse.ArgumentParser(description="Qdrant Semantic Search Script")

    # Arguments for Qdrant configuration
    parser.add_argument("--qdrant_host", type=str, required=True, help="Qdrant host URL.")
    parser.add_argument("--qdrant_api_key", type=str, required=True, help="API key for Qdrant access.")
    parser.add_argument("--index_name", required=True, help="Name of the collection in Qdrant.")
    parser.add_argument("--dimension", type=int, default=768, help="Dimension of the vectors in the collection.")
    parser.add_argument("--metric", type=str, default="Cosine", help="Distance metric for the collection (default: Cosine).")

    # Arguments for embedding models
    parser.add_argument(
        "--dense_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Dense embedding model."
    )
    parser.add_argument("--question", type=str, required=True, help="Question to perform semantic search on.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to retrieve.")

    args = parser.parse_args()

    # Initialize Qdrant vector DB client
    qdrant_client = QdrantClient(url=args.qdrant_host, api_key=args.qdrant_api_key)


    # Initialize dense embedding model
    text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    text_embedder.warm_up()

    # Generate dense embedding for the question
    question_embedding = text_embedder.run(text=args.question)["embedding"]

    # Perform query on Qdrant
    query_response = qdrant_client.search(
        collection_name=args.index_name,
        query_vector=question_embedding,
        limit=args.top_k,
    )

    # Convert query results to Haystack documents and print
    retrieval_results = QdrantDocumentConverter.convert_query_results_to_haystack_documents(query_response)
    print(retrieval_results)


if __name__ == "__main__":
    main()

