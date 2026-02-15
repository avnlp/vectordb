"""Pinecone namespaces management script.

This module provides functionality to manage namespaces
in Pinecone vector database using Haystack components.
"""

import argparse

from dataloaders.llms.groq import ChatGroqGenerator
from dataloaders.triviaqa_dataloader import TriviaQADataloader
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)
from pinecone import ServerlessSpec

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    """Manage Pinecone namespaces with hybrid embeddings.

    This function loads data, generates dense and sparse embeddings,
    upserts them into separate namespaces, and queries both.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Hybrid embedding upsert and retrieval using Pinecone."
    )

    # Pinecone VectorDB parameters
    parser.add_argument(
        "--api_key", required=True, help="API key for accessing Pinecone."
    )
    parser.add_argument(
        "--index_name", required=True, help="Name of the Pinecone index."
    )
    parser.add_argument(
        "--namespace1", required=True, help="Namespace for the first data split."
    )
    parser.add_argument(
        "--namespace2", required=True, help="Namespace for the second data split."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Vector dimension for the Pinecone index.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Similarity metric to use in the Pinecone index.",
    )
    parser.add_argument(
        "--cloud", default="aws", help="Cloud provider hosting the Pinecone database."
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="Region where the Pinecone index is hosted.",
    )
    parser.add_argument(
        "--tracing_project_name", type=str, help="Name of the tracing project."
    )
    parser.add_argument(
        "--weave_params",
        type=str,
        help="JSON string of parameters for configuring Weave.",
    )

    # Dataloader parameters
    parser.add_argument(
        "--dataset_name", required=True, help="Name of the dataset to use."
    )
    parser.add_argument(
        "--split1",
        required=True,
        help="Dataset split for the first namespace (e.g., 'train').",
    )
    parser.add_argument(
        "--split2",
        required=True,
        help="Dataset split for the second namespace (e.g., 'test').",
    )

    # Embedding model parameters
    parser.add_argument(
        "--dense_model", type=str, required=True, help="Dense embedding model name."
    )
    parser.add_argument(
        "--sparse_model", type=str, required=True, help="Sparse embedding model name."
    )

    # Query parameters
    parser.add_argument(
        "--question", type=str, required=True, help="Query/question text."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve."
    )

    args = parser.parse_args()

    # Load data splits using TriviaQADataloader
    dataloader_1 = TriviaQADataloader(
        answer_summary_generator=ChatGroqGenerator,
        dataset_name=args.dataset_name,
        split=args.split1,
    )
    dataloader_2 = TriviaQADataloader(
        answer_summary_generator=ChatGroqGenerator,
        dataset_name=args.dataset_name,
        split=args.split2,
    )
    dataloader_1.load_data()
    dataloader_2.load_data()

    haystack_documents_1 = dataloader_1.get_haystack_documents()
    haystack_documents_2 = dataloader_2.get_haystack_documents()

    # Initialize embedders
    dense_embedder = SentenceTransformersDocumentEmbedder(model=args.dense_model)
    dense_embedder.warm_up()
    sparse_embedder = FastembedSparseDocumentEmbedder(model=args.sparse_model)
    sparse_embedder.warm_up()

    # Generate embeddings for both splits
    split_1_docs_with_dense_embeddings = dense_embedder.run(
        documents=haystack_documents_1
    )["documents"]
    split_1_docs_with_sparse_embeddings = sparse_embedder.run(
        documents=split_1_docs_with_dense_embeddings
    )["documents"]

    split_2_docs_with_dense_embeddings = dense_embedder.run(
        documents=haystack_documents_2
    )["documents"]
    split_2_docs_with_sparse_embeddings = sparse_embedder.run(
        documents=split_2_docs_with_dense_embeddings
    )["documents"]

    # Initialize Pinecone and create index
    pinecone_vector_db = PineconeVectorDB(api_key=args.api_key)
    pinecone_vector_db.create_index(
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        spec=ServerlessSpec(cloud=args.cloud, region=args.region),
    )

    # Prepare data for upsert
    docs_for_pinecone_split_1 = (
        PineconeDocumentConverter.prepare_haystack_documents_for_upsert(
            split_1_docs_with_sparse_embeddings
        )
    )
    docs_for_pinecone_split_2 = (
        PineconeDocumentConverter.prepare_haystack_documents_for_upsert(
            split_2_docs_with_sparse_embeddings
        )
    )

    # Upsert data into namespaces
    pinecone_vector_db.upsert(
        data=docs_for_pinecone_split_1,
        namespace=args.namespace1,
    )
    pinecone_vector_db.upsert(
        data=docs_for_pinecone_split_2,
        namespace=args.namespace2,
    )

    # Query Pinecone for both namespaces
    dense_text_embedder = SentenceTransformersTextEmbedder(model=args.dense_model)
    dense_text_embedder.warm_up()
    sparse_text_embedder = FastembedSparseTextEmbedder(model=args.sparse_model)
    sparse_text_embedder.warm_up()

    dense_question_embedding = dense_text_embedder.run(text=args.question)["embedding"]
    sparse_question_embedding = sparse_text_embedder.run(text=args.question)[
        "sparse_embedding"
    ].to_dict()

    # Query namespace 1
    query_response_split_1 = pinecone_vector_db.query(
        vector=dense_question_embedding,
        sparse_vector=sparse_question_embedding,
        top_k=args.top_k,
        namespace=args.namespace1,
    )
    retrieval_results_split_1 = (
        PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_response_split_1
        )
    )
    print("Results from namespace 1:")
    for result in retrieval_results_split_1:
        print(result)

    # Query namespace 2
    query_response_split_2 = pinecone_vector_db.query(
        vector=dense_question_embedding,
        sparse_vector=sparse_question_embedding,
        top_k=args.top_k,
        namespace=args.namespace2,
    )
    retrieval_results_split_2 = (
        PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_response_split_2
        )
    )
    print("Results from namespace 2:")
    for result in retrieval_results_split_2:
        print(result)


if __name__ == "__main__":
    main()
