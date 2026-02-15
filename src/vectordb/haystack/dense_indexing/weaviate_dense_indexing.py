import argparse
from ast import literal_eval

from dataloaders import (
    ARCDataloader,
    EdgarDataloader,
    FactScoreDataloader,
    PopQADataloader,
    TriviaQADataloader,
)
from dataloaders.llms import ChatGroqGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from vectordb import WeaviateDocumentConverter, WeaviateVectorDB


def main():
    """Process data, generate embeddings, and index them into a Weaviate vector database.

    This script is designed to:
    - Load datasets using specified dataloaders.
    - Optionally use an LLM-based generator for answer summarization.
    - Split and preprocess text data.
    - Generate embeddings for documents.
    - Index processed data into a Weaviate vector database.
    """
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(
        description="Script for processing and indexing data with Weaviate."
    )

    # Dataloader parameters
    parser.add_argument(
        "--dataloader",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Dataloader to use for loading datasets.",
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset to be used by the dataloader.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to process (e.g., 'test', 'train').",
    )
    parser.add_argument(
        "--text_splitter",
        default="RecursiveCharacterTextSplitter",
        help="Text splitter method to preprocess documents.",
    )
    parser.add_argument(
        "--text_splitter_params",
        type=str,
        help="JSON string of parameters for configuring the text splitter.",
    )

    # Generator parameters
    parser.add_argument(
        "--generator_model", type=str, help="Model name for the dataloader's generator."
    )
    parser.add_argument(
        "--generator_api_key", help="API key for the dataloader generator."
    )
    parser.add_argument(
        "--generator_llm_params",
        type=str,
        help="JSON string of parameters for the generator LLM.",
    )

    # Embedder parameters
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to use for generating document embeddings.",
    )
    parser.add_argument(
        "--embedding_model_params",
        type=str,
        help="JSON string of parameters for the embedding model.",
    )

    # Weaviate VectorDB parameters
    parser.add_argument(
        "--weaviate_cluster_url", required=True, help="Weaviate cluster URL."
    )
    parser.add_argument(
        "--weaviate_api_key", required=True, help="API key for accessing Weaviate."
    )
    parser.add_argument(
        "--headers", type=str, help="JSON string of headers for the Weaviate API."
    )
    parser.add_argument(
        "--collection_name", type=str, help="Name of the collection to interact with."
    )
    parser.add_argument(
        "--tracing_project_name",
        default="weaviate",
        help="Name of the Weave project for tracing.",
    )
    parser.add_argument(
        "--weave_params", type=str, help="JSON string of additional Weave parameters."
    )

    args = parser.parse_args()

    # Parse parameters
    text_splitter_params = (
        literal_eval(args.text_splitter_params) if args.text_splitter_params else {}
    )
    generator_params = (
        literal_eval(args.generator_llm_params) if args.generator_llm_params else {}
    )
    embedding_model_params = (
        literal_eval(args.embedding_model_params) if args.embedding_model_params else {}
    )
    headers = literal_eval(args.headers) if args.headers else {}
    weave_params = literal_eval(args.weave_params) if args.weave_params else {}

    # Instantiate generator if model and API key are provided
    generator = None
    if args.generator_model and args.generator_api_key:
        generator = ChatGroqGenerator(
            model=args.generator_model,
            api_key=args.generator_api_key,
            llm_params=generator_params,
        )

    # Map the dataloader name to its corresponding class
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }
    dataloader_cls = dataloader_map[args.dataloader]
    dataloader_kwargs = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "text_splitter": args.text_splitter,
        "text_splitter_params": text_splitter_params,
    }
    if generator:
        dataloader_kwargs["answer_summary_generator"] = generator
    dataloader = dataloader_cls(**dataloader_kwargs)

    # Load data and preprocess
    dataloader.load_data()
    haystack_documents = dataloader.get_haystack_documents()

    # Load the embedding model
    embedder = SentenceTransformersDocumentEmbedder(
        model=args.embedding_model, **embedding_model_params
    )
    embedder.warm_up()

    # Create the embeddings for each document
    docs_with_embeddings = embedder.run(documents=haystack_documents)["documents"]

    # Initialize Weaviate VectorDB
    weaviate_vectordb = WeaviateVectorDB(
        cluster_url=args.weaviate_cluster_url,
        api_key=args.weaviate_api_key,
        headers=headers,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )

    # Create the collection in Weaviate
    weaviate_vectordb.create_collection(collection_name=args.collection_name)

    # Prepare and upsert the documents to Weaviate
    data_for_weaviate = WeaviateDocumentConverter.prepare_haystack_documents_for_upsert(
        docs_with_embeddings
    )
    weaviate_vectordb.upsert(data=data_for_weaviate)


if __name__ == "__main__":
    main()
