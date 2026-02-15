"""Weaviate dense indexing example with LangChain integration.

This module demonstrates how to perform dense indexing using Weaviate VectorDB
with HuggingFace embeddings and various datasets.
"""

import argparse

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
    """Run the Weaviate dense indexing pipeline.

    This function:
    - Parses command line arguments
    - Initializes the data loader and generator
    - Loads and processes the dataset
    - Generates embeddings using HuggingFace embeddings
    - Creates a Weaviate collection and upserts the data
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Load data, generate embeddings, and upsert to Weaviate."
    )

    # Add arguments
    parser.add_argument(
        "--dataloader_name",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Name of the dataloader to use.",
    )
    parser.add_argument(
        "--dataset_name", default="awinml/triviaqa", help="Name of the dataset."
    )
    parser.add_argument("--split", default="test[:5]", help="Dataset split to use.")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model name.",
    )
    parser.add_argument("--weaviate_url", required=True, help="Weaviate cluster URL.")
    parser.add_argument(
        "--weaviate_api_key", required=True, help="API key for Weaviate."
    )
    parser.add_argument(
        "--collection_name", required=True, help="Name of the collection in Weaviate."
    )

    args = parser.parse_args()

    # Initialize dataloader
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }
    dataloader_cls = dataloader_map[args.dataloader_name]

    generator = ChatGroqGenerator(
        model="llama-3.1-8b-instant",
        api_key="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",
        llm_params={
            "temperature": 0,
            "max_tokens": 1024,
            "timeout": 360,
            "max_retries": 100,
        },
    )

    dataloader = dataloader_cls(
        dataset_name=args.dataset_name,
        split=args.split,
        answer_summary_generator=generator,
    )

    # Load data
    dataloader.load_data()
    dataloader.get_questions()
    haystack_documents = dataloader.get_haystack_documents()

    # Initialize embedder
    embedder = SentenceTransformersDocumentEmbedder(model=args.model)
    embedder.warm_up()

    # Create embeddings
    docs_with_embeddings = embedder.run(documents=haystack_documents)["documents"]

    # Initialize Weaviate vector database
    weaviate_vectordb = WeaviateVectorDB(
        cluster_url=args.weaviate_url, api_key=args.weaviate_api_key
    )

    # Create the collection in Weaviate
    weaviate_vectordb.create_collection(collection_name=args.collection_name)

    # Prepare data and upsert to Weaviate
    data_for_weaviate = WeaviateDocumentConverter.prepare_haystack_documents_for_upsert(
        docs_with_embeddings
    )
    weaviate_vectordb.upsert(data=data_for_weaviate)

    print(
        f"Upserted documents to Weaviate collection '{args.collection_name}' successfully."
    )


if __name__ == "__main__":
    main()
