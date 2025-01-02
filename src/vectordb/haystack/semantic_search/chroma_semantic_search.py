import argparse

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder

from vectordb import ChromaDocumentConverter, ChromaVectorDB


def main():
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Script for processing and indexing data with Chroma.")

    # Dataloader parameters
    parser.add_argument(
        "--dataloader",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Dataloader to use for loading datasets.",
    )
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset to be used by the dataloader.")
    parser.add_argument("--split", default="test", help="Dataset split to process (e.g., 'test', 'train').")
    parser.add_argument(
        "--text_splitter",
        default="RecursiveCharacterTextSplitter",
        help="Text splitter method to preprocess documents.",
    )
    parser.add_argument(
        "--text_splitter_params", type=str, help="JSON string of parameters for configuring the text splitter."
    )

    # Generator parameters
    parser.add_argument("--generator_model", type=str, help="Model name for the dataloader's generator.")
    parser.add_argument("--generator_api_key", help="API key for the dataloader generator.")
    parser.add_argument("--generator_llm_params", type=str, help="JSON string of parameters for the generator LLM.")

    # Embedder parameters
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to use for generating document embeddings.",
    )
    parser.add_argument("--embedding_model_params", type=str, help="JSON string of parameters for the embedding model.")

    # Chroma VectorDB arguments
    parser.add_argument("--chroma_path", default="./chroma_database_files", help="Path for Chroma database files.")
    parser.add_argument("--collection_name", default="test_collection_dense1", help="Name of the Chroma collection.")
    parser.add_argument("--tracing_project_name", default="chroma", help="Name of the Weave project for tracing.")
    parser.add_argument("--weave_params", type=str, help="JSON string of parameters for initializing Weave.")

    # Query Arguments
    parser.add_argument("--question", required=True, help="Question to query the vector database.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to retrieve.")

    args = parser.parse_args()

    # Initialize Chroma VectorDB
    chroma_vector_db = ChromaVectorDB(persistent=True, path=args.chroma_path)
    chroma_vector_db.create_collection(name=args.chroma_collection)

    # Initialize Text Embedders
    dense_text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    dense_text_embedder.warm_up()

    sparse_text_embedder = FastembedSparseTextEmbedder(model=args.sparse_model)
    sparse_text_embedder.warm_up()

    # Generate Dense and Sparse Embeddings
    dense_question_embedding = dense_text_embedder.run(text=args.question)["embedding"]
    sparse_question_embedding = sparse_text_embedder.run(text=args.question)["sparse_embedding"].to_dict()

    # Query the Chroma VectorDB
    query_response = chroma_vector_db.query(
        vector=dense_question_embedding,
        sparse_vector=sparse_question_embedding,
        top_k=args.top_k,
        include_metadata=args.include_metadata,
        namespace=args.namespace,
    )

    # Convert query results to Haystack documents
    retrieval_results = ChromaDocumentConverter.convert_query_results_to_haystack_documents(query_response)
    print(retrieval_results)


if __name__ == "__main__":
    main()
