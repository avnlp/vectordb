"""Metadata filtering script for Weaviate vector database.

This module provides functionality to query Weaviate vector database
with metadata filtering using Haystack components.
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
from haystack.components.embedders import SentenceTransformersTextEmbedder
from weaviate.classes.query import Filter

from vectordb import WeaviateDocumentConverter, WeaviateVectorDB


def main():
    """Perform Metadata Filtering using Weaviate."""
    # Argument parser for user inputs
    parser = argparse.ArgumentParser(
        description="Script for querying Weaviate with metadata filtering."
    )

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
    literal_eval(args.embedding_model_params) if args.embedding_model_params else {}
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
    dataloader_cls(**dataloader_kwargs)

    # Instantiate the text embedder
    text_embedder = SentenceTransformersTextEmbedder(model=args.embedding_model)
    text_embedder.warm_up()

    # Get the embedding for the question
    dense_question_embedding = text_embedder.run(text=args.question)["embedding"]

    # Prepare the filter for querying metadata
    filters = Filter.by_property(args.filter_property).like(args.filter_value)

    # Initialize Weaviate VectorDB
    weaviate_vector_db = WeaviateVectorDB(
        cluster_url=args.weaviate_cluster_url,
        api_key=args.weaviate_api_key,
        headers=headers,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )

    # Query the Weaviate VectorDB
    query_response = weaviate_vector_db.query(
        vector=dense_question_embedding,
        query_string=args.question,
        limit=10,
        hybrid=True,
        alpha=0.5,
        filters=filters,
    )

    # Convert the query results to Haystack documents
    retrieval_results = (
        WeaviateDocumentConverter.convert_query_results_to_haystack_documents(
            query_response
        )
    )
    print(retrieval_results)


if __name__ == "__main__":
    main()
