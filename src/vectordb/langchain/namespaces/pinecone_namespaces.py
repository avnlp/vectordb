"""Pinecone namespaces example with LangChain integration.

This module demonstrates how to work with Pinecone namespaces for organizing
and querying vector data across multiple data splits.
"""

import argparse

from dataloaders.llms.groq import ChatGroqGenerator
from dataloaders.triviaqa_dataloader import TriviaQADataloader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from pinecone import ServerlessSpec

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    """Run the Pinecone namespaces example.

    This function:
    - Parses command line arguments
    - Loads data from two different splits
    - Generates embeddings for each split
    - Creates a Pinecone index
    - Upserts data into separate namespaces
    - Queries both namespaces and prints results
    """
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Process multiple data splits and query Pinecone namespaces."
    )

    # Arguments for data splits
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")
    parser.add_argument(
        "--split1", required=True, help="Data split for the first namespace."
    )
    parser.add_argument(
        "--split2", required=True, help="Data split for the second namespace."
    )

    # Embedding models
    parser.add_argument(
        "--dense_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model.",
    )
    parser.add_argument(
        "--sparse_model",
        default="prithivida/Splade_PP_en_v1",
        help="Sparse embedding model.",
    )

    # Pinecone configurations
    parser.add_argument("--pinecone_api_key", required=True, help="Pinecone API key.")
    parser.add_argument(
        "--index_name", default="test-index-namespaces", help="Pinecone index name."
    )
    parser.add_argument(
        "--dimension", type=int, default=768, help="Embedding dimension."
    )
    parser.add_argument("--metric", default="dotproduct", help="Similarity metric.")
    parser.add_argument("--cloud", default="aws", help="Cloud provider for Pinecone.")
    parser.add_argument("--region", default="us-east-1", help="Region for Pinecone.")
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for upserts."
    )
    parser.add_argument(
        "--namespace1", default="test_namespace1", help="Namespace for split1."
    )
    parser.add_argument(
        "--namespace2", default="test_namespace2", help="Namespace for split2."
    )

    # Query configurations
    parser.add_argument(
        "--query", required=True, help="Query to run against the namespaces."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of top results to retrieve."
    )

    args = parser.parse_args()

    # Load dataloaders
    dataloader1 = TriviaQADataloader(
        answer_summary_generator=ChatGroqGenerator,
        dataset_name=args.dataset_name,
        split=args.split1,
    )
    dataloader2 = TriviaQADataloader(
        answer_summary_generator=ChatGroqGenerator,
        dataset_name=args.dataset_name,
        split=args.split2,
    )

    langchain_documents1 = dataloader1.get_langchain_documents()
    langchain_documents2 = dataloader2.get_langchain_documents()

    # Load embedding models
    embedder = HuggingFaceEmbeddings(model_name=args.dense_model)
    sparse_embedder = FastEmbedSparse(model_name=args.sparse_model)

    # Generate embeddings
    texts1 = [doc.page_content for doc in langchain_documents1]
    doc_embeddings1 = embedder.embed_documents(texts1)
    sparse_embeddings1 = sparse_embedder.embed_documents(texts1)

    texts2 = [doc.page_content for doc in langchain_documents2]
    doc_embeddings2 = embedder.embed_documents(texts2)
    sparse_embeddings2 = sparse_embedder.embed_documents(texts2)

    # Initialize Pinecone
    pinecone_vector_db = PineconeVectorDB(api_key=args.pinecone_api_key)
    pinecone_vector_db.create_index(
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        spec=ServerlessSpec(cloud=args.cloud, region=args.region),
    )

    # Prepare and upsert documents for namespaces
    docs_split1 = PineconeDocumentConverter.prepare_langchain_documents_for_upsert(
        documents=langchain_documents1,
        embeddings=doc_embeddings1,
        sparse_embeddings=sparse_embeddings1,
    )
    docs_split2 = PineconeDocumentConverter.prepare_langchain_documents_for_upsert(
        documents=langchain_documents2,
        embeddings=doc_embeddings2,
        sparse_embeddings=sparse_embeddings2,
    )

    pinecone_vector_db.upsert(
        data=docs_split1,
        namespace=args.namespace1,
        batch_size=args.batch_size,
        show_progress=True,
    )
    pinecone_vector_db.upsert(
        data=docs_split2,
        namespace=args.namespace2,
        batch_size=args.batch_size,
        show_progress=True,
    )

    # Query namespaces
    dense_query_embedding = embedder.embed_query(args.query)
    sparse_query_embedding = sparse_embedder.embed_query(args.query)

    query_response1 = pinecone_vector_db.query(
        vector=dense_query_embedding,
        sparse_vector={
            "indices": sparse_query_embedding.indices,
            "values": sparse_query_embedding.values,
        },
        top_k=args.top_k,
        include_metadata=True,
        namespace=args.namespace1,
    )
    results_split1 = (
        PineconeDocumentConverter.convert_query_results_to_langchain_documents(
            query_response1
        )
    )

    query_response2 = pinecone_vector_db.query(
        vector=dense_query_embedding,
        sparse_vector={
            "indices": sparse_query_embedding.indices,
            "values": sparse_query_embedding.values,
        },
        top_k=args.top_k,
        include_metadata=True,
        namespace=args.namespace2,
    )
    results_split2 = (
        PineconeDocumentConverter.convert_query_results_to_langchain_documents(
            query_response2
        )
    )

    print("Results from namespace 1:")
    print(results_split1)

    print("Results from namespace 2:")
    print(results_split2)


if __name__ == "__main__":
    main()
