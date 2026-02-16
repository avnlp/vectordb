"""Dense indexing script for Chroma vector database.

This module provides functionality to process data, generate embeddings,
and index them into a Chroma vector database using Haystack components.
"""

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

from vectordb import ChromaDocumentConverter, ChromaVectorDB


def main():
    """Process data, generate embeddings, and index them into a Chroma vector database.

    This script is designed to:
    - Load datasets using specified dataloaders.
    - Optionally use an LLM-based generator for answer summarization.
    - Split and preprocess text data.
    - Generate embeddings for documents.
    - Index processed data into a Chroma vector database.
    """
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(
        description="Script for processing and indexing data with Chroma."
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

    # Chroma VectorDB arguments
    parser.add_argument(
        "--chroma_path",
        default="./chroma_database_files",
        help="Path for Chroma database files.",
    )
    parser.add_argument(
        "--collection_name",
        default="test_collection_dense1",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--tracing_project_name",
        default="chroma",
        help="Name of the Weave project for tracing.",
    )
    parser.add_argument(
        "--weave_params",
        type=str,
        help="JSON string of parameters for initializing Weave.",
    )

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
    dataloader = dataloader_cls(
        dataset_name=args.dataset_name,
        split=args.split,
        answer_summary_generator=generator,
    )

    # Load the data
    dataloader.load_data()
    dataloader.get_questions()
    haystack_documents = dataloader.get_haystack_documents()

    # Load the embedding model
    embedder = SentenceTransformersDocumentEmbedder(model=args.embedding_model)
    embedder.warm_up()

    # Create the embeddings for each document
    docs_with_embeddings = embedder.run(documents=haystack_documents)["documents"]

    # Initialize Chroma VectorDB
    weave_params = literal_eval(args.weave_params) if args.weave_params else {}
    chroma_vector_db = ChromaVectorDB(
        persistent=True,
        path=args.chroma_path,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )
    chroma_vector_db.create_collection(name=args.collection_name)

    # Add the documents to Chroma
    data_for_chroma = ChromaDocumentConverter.prepare_haystack_documents_for_upsert(
        docs_with_embeddings
    )
    chroma_vector_db.upsert(data=data_for_chroma)


if __name__ == "__main__":
    main()
