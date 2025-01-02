import argparse

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from vectordb import ChromaDocumentConverter, ChromaVectorDB


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Query Chroma VectorDB with a question")

    # Arguments for the Chroma database
    parser.add_argument("--chroma_path", type=str, required=True, help="Path to Chroma database files")
    parser.add_argument("--collection_name", type=str, required=True, help="Name of the Chroma collection")

    # Arguments for embeddings and querying
    parser.add_argument(
        "--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Model name for embeddings"
    )
    parser.add_argument("--question", type=str, required=True, help="The question to query the Chroma database with")

    args = parser.parse_args()

    # Initialize ChromaVectorDB
    chroma_vector_db = ChromaVectorDB(path=args.chroma_path)
    chroma_vector_db.create_collection(name=args.collection_name)

    # Initialize the text embedder
    text_embedder = HuggingFaceEmbeddings(model_name=args.model_name)

    # Embed the question
    dense_question_embedding = text_embedder.embed_query(args.question)

    # Query Chroma VectorDB using the filter
    query_response = chroma_vector_db.query(
        query_embedding=dense_question_embedding,
        n_results=10,
        filter={
            "$and": [
                {"$contains": {"field_name": "sampras"}},  # Adjust field_name based on your metadata
                {"$contains": {"field_name": "tennis"}},
            ]
        },
    )

    print(query_response)

    # Convert query results to Langchain documents
    retrieval_results = ChromaDocumentConverter.convert_query_results_to_langchain_documents(query_response)
    print(retrieval_results)


if __name__ == "__main__":
    main()
