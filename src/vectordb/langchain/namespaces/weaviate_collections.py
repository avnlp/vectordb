"""Weaviate collections example with LangChain integration.

This module demonstrates how to work with Weaviate collections for organizing
and querying vector data across multiple collections.
"""

import argparse

from dataloaders.llms.groq import ChatGroqGenerator
from dataloaders.triviaqa_dataloader import TriviaQADataloader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from vectordb import WeaviateDocumentConverter, WeaviateVectorDB


def parse_args():
    """Parse command line arguments for Weaviate collections example.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Upsert documents to Weaviate and query the database."
    )
    parser.add_argument(
        "--cluster_url",
        type=str,
        required=True,
        help="The URL of the Weaviate cluster.",
    )
    parser.add_argument(
        "--api_key", type=str, required=True, help="API key for accessing Weaviate."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="awinml/triviaqa",
        help="The dataset name for TriviaQA.",
    )
    parser.add_argument(
        "--split_1",
        type=str,
        default="test[:10]",
        help="The split for the first set of documents.",
    )
    parser.add_argument(
        "--split_2",
        type=str,
        default="test[10:20]",
        help="The split for the second set of documents.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="The embedding model.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The question to query Weaviate with.",
    )
    parser.add_argument(
        "--collection_name_1",
        type=str,
        default="test_collection_dense_split_1",
        help="Collection name for split 1.",
    )
    parser.add_argument(
        "--collection_name_2",
        type=str,
        default="test_collection_dense_split_2",
        help="Collection name for split 2.",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="The number of results to retrieve."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value for hybrid query."
    )

    return parser.parse_args()


def main():
    """Run the Weaviate collections example.

    This function:
    - Parses command line arguments
    - Loads data from two different splits
    - Generates embeddings for each split
    - Creates Weaviate collections
    - Upserts data into separate collections
    - Queries both collections and prints results
    """
    # Parse arguments
    args = parse_args()

    # Initialize the dataloaders
    dataloader_1 = TriviaQADataloader(
        answer_summary_generator=ChatGroqGenerator,
        dataset_name=args.dataset_name,
        split=args.split_1,
    )

    dataloader_2 = TriviaQADataloader(
        answer_summary_generator=ChatGroqGenerator,
        dataset_name=args.dataset_name,
        split=args.split_2,
    )

    # Load the data
    dataloader_1.load_data()
    langchain_documents_1 = dataloader_1.get_langchain_documents()

    dataloader_2.load_data()
    langchain_documents_2 = dataloader_2.get_langchain_documents()

    # Load the embedding model
    embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)

    # Create the embeddings for each document
    texts_split_1 = [doc.page_content for doc in langchain_documents_1]
    doc_embeddings_split_1 = embedder.embed_documents(texts_split_1)

    texts_split_2 = [doc.page_content for doc in langchain_documents_2]
    doc_embeddings_split_2 = embedder.embed_documents(texts_split_2)

    data_for_weaviate_split_1 = (
        WeaviateDocumentConverter.prepare_langchain_documents_for_upsert(
            documents=langchain_documents_1, embeddings=doc_embeddings_split_1
        )
    )
    data_for_weaviate_split_2 = (
        WeaviateDocumentConverter.prepare_langchain_documents_for_upsert(
            documents=langchain_documents_2, embeddings=doc_embeddings_split_2
        )
    )

    # Initialize WeaviateVectorDB
    weaviate_vectordb = WeaviateVectorDB(
        cluster_url=args.cluster_url,
        api_key=args.api_key,
    )

    # Create the collections in Weaviate
    weaviate_vectordb.create_collection(collection_name=args.collection_name_1)
    weaviate_vectordb.upsert(data=data_for_weaviate_split_1)

    weaviate_vectordb.create_collection(collection_name=args.collection_name_2)
    weaviate_vectordb.upsert(data=data_for_weaviate_split_2)

    # Query Weaviate
    dense_question_embedding = embedder.embed_query(args.question)

    query_response_split_1 = weaviate_vectordb.query(
        vector=dense_question_embedding,
        limit=args.limit,
        hybrid=True,
        alpha=args.alpha,
        query_string=args.question,
    )
    retrieval_results_split_1 = (
        WeaviateDocumentConverter.convert_query_results_to_langchain_documents(
            query_response_split_1
        )
    )
    print("Results for Split 1:", retrieval_results_split_1)

    query_response_split_2 = weaviate_vectordb.query(
        vector=dense_question_embedding,
        limit=args.limit,
        hybrid=True,
        alpha=args.alpha,
        query_string=args.question,
    )
    retrieval_results_split_2 = (
        WeaviateDocumentConverter.convert_query_results_to_langchain_documents(
            query_response_split_2
        )
    )
    print("Results for Split 2:", retrieval_results_split_2)


if __name__ == "__main__":
    main()
