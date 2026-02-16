"""Weaviate collections management script.

This module provides functionality to manage multiple collections
in Weaviate vector database using Haystack components.
"""

import argparse
from ast import literal_eval

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from weaviate import Filter

from vectordb import WeaviateDocumentConverter, WeaviateVectorDB


def main():
    """Main function to handle Weaviate indexing and querying."""
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

    # Weaviate parameters
    parser.add_argument(
        "--weaviate_cluster_url", required=True, help="Weaviate cluster URL."
    )
    parser.add_argument(
        "--weaviate_api_key", required=True, help="API key for accessing Weaviate."
    )
    parser.add_argument(
        "--collection_name1", type=str, required=True, help="First collection name."
    )
    parser.add_argument(
        "--collection_name2", type=str, required=True, help="Second collection name."
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Query string for Weaviate search."
    )

    # Parse arguments
    args = parser.parse_args()

    # Parse JSON strings
    text_splitter_params = (
        literal_eval(args.text_splitter_params) if args.text_splitter_params else {}
    )
    generator_params = (
        literal_eval(args.generator_llm_params) if args.generator_llm_params else {}
    )
    embedding_model_params = (
        literal_eval(args.embedding_model_params) if args.embedding_model_params else {}
    )

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
    embedder = SentenceTransformersDocumentEmbedder(
        model=args.embedding_model, **embedding_model_params
    )
    embedder.warm_up()

    # Create embeddings for documents
    docs_with_embeddings = embedder.run(documents=haystack_documents)["documents"]

    # Initialize Weaviate VectorDB
    weaviate_vectordb = WeaviateVectorDB(
        cluster_url=args.weaviate_cluster_url, api_key=args.weaviate_api_key
    )

    # Prepare documents for Weaviate upsert
    docs_for_weaviate = WeaviateDocumentConverter.prepare_haystack_documents_for_upsert(
        docs_with_embeddings
    )

    # Create collections in Weaviate
    weaviate_vectordb.create_collection(collection_name=args.collection_name1)
    weaviate_vectordb.create_collection(collection_name=args.collection_name2)

    # Upsert data into Weaviate collections
    weaviate_vectordb.upsert(data=docs_for_weaviate, collection=args.collection_name1)
    weaviate_vectordb.upsert(data=docs_for_weaviate, collection=args.collection_name2)

    # Query data from Weaviate
    dense_question_embedding = embedder.run(text=args.query)["embedding"]
    query_response = weaviate_vectordb.query(
        vector=dense_question_embedding,
        query_string=args.query,
        limit=10,
        hybrid=True,
        alpha=0.5,
        filters=Filter.by_property("text").like("Chipmunks"),
    )

    # Convert query results to Haystack documents
    retrieval_results = (
        WeaviateDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )
    print(retrieval_results)


if __name__ == "__main__":
    main()
