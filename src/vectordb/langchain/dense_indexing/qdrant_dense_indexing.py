import argparse
from ast import literal_eval

from dataloaders import ARCDataloader, EdgarDataloader, FactScoreDataloader, PopQADataloader, TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct


def main():
    """Process data, generate embeddings, and index them into a Qdrant vector database.

    This script is designed to:
    - Load datasets using specified dataloaders.
    - Optionally use an LLM-based generator for answer summarization.
    - Split and preprocess text data.
    - Generate embeddings for documents.
    - Index processed data into a Qdrant vector database.
    """
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

    # Qdrant VectorDB arguments
    parser.add_argument("--qdrant_host", default="localhost", help="Host for Qdrant server.")
    parser.add_argument("--qdrant_port", type=int, default=6333, help="Port for Qdrant server.")
    parser.add_argument("--collection_name", default="test_collection_dense1", help="Name of the Qdrant collection.")
    parser.add_argument("--vector_size", type=int, required=True, help="Dimensionality of the vector embeddings.")

    args = parser.parse_args()

    # Initialize the generator
    generator_params = literal_eval(args.generator_llm_params)
    generator = ChatGroqGenerator(
        model=args.generator_model,
        api_key=args.generator_api_key,
        llm_params=generator_params,
    )

    # Map dataloader names to classes
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }

    dataloader_cls = dataloader_map[args.dataloader]
    dataloader = dataloader_cls(dataset_name=args.dataset_name, split=args.split, answer_summary_generator=generator)

    # Load the data
    dataloader.load_data()
    dataloader.get_questions()
    langchain_documents = dataloader.get_langchain_documents()

    # Load the embedding model
    embedder = HuggingFaceEmbeddings(model=args.embedding_model)

    # Create the embeddings for each document
    docs_with_embeddings = embedder.run(documents=langchain_documents)["documents"]

    # Initialize Qdrant client
    qdrant_client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)

    # Create or verify the collection in Qdrant
    collection_name = args.collection_name
    vector_size = args.vector_size

    if not qdrant_client.get_collection(collection_name, raise_on_not_found=False):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine"),
        )
    else:
        print(f"Collection '{collection_name}' already exists.")

    # Prepare data for upsert into Qdrant
    points = [
        PointStruct(
            id=doc.meta["id"],
            vector=doc.embedding,
            payload={"text": doc.content, **doc.meta},
        )
        for doc in docs_with_embeddings
    ]

    # Upsert data into Qdrant
    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Indexed {len(points)} documents into Qdrant collection '{collection_name}'.")


if __name__ == "__main__":
    main()

