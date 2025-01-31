import argparse
from ast import literal_eval

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from vectordb import QdrantDocumentConverter, QdrantVectorDB


def main():
    """Main function to handle Qdrant indexing and querying."""
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Script for processing and indexing data with Qdrant.")
    
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

    # Qdrant parameters
    parser.add_argument("--qdrant_host", required=True, help="Qdrant host URL.")
    parser.add_argument("--qdrant_api_key", required=True, help="API key for accessing Qdrant.")
    parser.add_argument("--collection_name1", type=str, required=True, help="First collection name.")
    parser.add_argument("--collection_name2", type=str, required=True, help="Second collection name.")
    parser.add_argument("--query", type=str, required=True, help="Query string for Qdrant search.")

    # Parse arguments
    args = parser.parse_args()

    # Parse JSON strings
    text_splitter_params = literal_eval(args.text_splitter_params) if args.text_splitter_params else {}
    generator_params = literal_eval(args.generator_llm_params) if args.generator_llm_params else {}
    embedding_model_params = literal_eval(args.embedding_model_params) if args.embedding_model_params else {}

    # Initialize generator if model and API key are provided
    generator = None
    if args.generator_model and args.generator_api_key:
        generator = ChatGroqGenerator(
            model=args.generator_model,
            api_key=args.generator_api_key,
            llm_params=generator_params,
        )

    # Initialize dataloader
    dataloader = TriviaQADataloader(
        answer_summary_generator=generator,
        dataset_name=args.dataset_name,
        split=args.split,
        text_splitter=args.text_splitter,
        text_splitter_params=text_splitter_params,
    )

    # Load data and preprocess
    dataloader.load_data()
    haystack_documents = dataloader.get_haystack_documents()

    # Initialize the embedding model
    embedder = SentenceTransformersDocumentEmbedder(model=args.embedding_model, **embedding_model_params)
    embedder.warm_up()

    # Create embeddings for documents
    docs_with_embeddings = embedder.run(documents=haystack_documents)["documents"]

    # Initialize Qdrant VectorDB
    qdrant_client = QdrantClient(url=args.qdrant_host, api_key=args.qdrant_api_key)

    # Create collections in Qdrant (if they do not already exist)
    qdrant_client.create_collection(
        collection_name=args.collection_name1,
        vector_size=docs_with_embeddings[0]['embedding'].shape[0],  # Assuming all embeddings are of the same size
        distance="Cosine"
    )
    qdrant_client.create_collection(
        collection_name=args.collection_name2,
        vector_size=docs_with_embeddings[0]['embedding'].shape[0],
        distance="Cosine"
    )

    # Prepare documents for Qdrant upsert
    docs_for_qdrant = QdrantDocumentConverter.prepare_haystack_documents_for_upsert(docs_with_embeddings)

    # Upsert data into Qdrant collections
    qdrant_client.upsert(
        collection_name=args.collection_name1,
        points=docs_for_qdrant
    )
    qdrant_client.upsert(
        collection_name=args.collection_name2,
        points=docs_for_qdrant
    )

    # Query data from Qdrant
    dense_question_embedding = embedder.run(text=args.query)["embedding"]
    query_response = qdrant_client.search(
        collection_name=args.collection_name1,
        query_vector=dense_question_embedding,
        limit=10
    )

    # Convert query results to Haystack documents
    retrieval_results = QdrantDocumentConverter.convert_query_results_to_haystack_documents(query_response)
    print(retrieval_results)


if __name__ == "__main__":
    main()

