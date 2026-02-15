"""Milvus RAG pipeline example with LangChain integration.

This module demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline
using Milvus VectorDB and LangChain components.
"""

import argparse

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


def main():
    """Run the Milvus RAG pipeline.

    This function:
    - Parses command line arguments
    - Connects to Milvus and creates/loads a collection
    - Initializes the data loader and generator
    - Loads the TriviaQA dataset
    - Processes questions through the RAG pipeline
    - Generates answers using retrieved context
    """
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="RAG pipeline for question answering using Milvus and ChatGroq."
    )

    # Dataloader arguments
    parser.add_argument(
        "--dataset_name", required=True, help="Dataset name for TriviaQA dataloader."
    )
    parser.add_argument(
        "--split",
        default="test[:5]",
        help="Dataset split to use (default: 'test[:5]').",
    )

    # LLM generator arguments
    parser.add_argument(
        "--llm_model",
        default="llama-3.1-8b-instant",
        help="Model name for the LLM generator.",
    )
    parser.add_argument(
        "--llm_api_key", required=True, help="API key for the LLM service."
    )
    parser.add_argument(
        "--llm_params",
        type=str,
        default='{"temperature": 0, "max_tokens": 1024, "timeout": 360, "max_retries": 100}',
        help="JSON string of LLM parameters.",
    )

    # Embedding arguments
    parser.add_argument(
        "--dense_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Dense embedding model name.",
    )
    parser.add_argument(
        "--sparse_model",
        default="prithivida/Splade_PP_en_v1",
        help="Sparse embedding model name.",
    )

    # Milvus arguments
    parser.add_argument("--milvus_host", default="localhost", help="Milvus host.")
    parser.add_argument("--milvus_port", default="19530", help="Milvus port.")
    parser.add_argument(
        "--collection_name", default="rag_collection", help="Milvus collection name."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to retrieve from Milvus.",
    )

    args = parser.parse_args()

    # Connect to Milvus
    connections.connect("default", host=args.milvus_host, port=args.milvus_port)

    # Create or load collection
    if not utility.has_collection(args.collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="vector", dtype=DataType.FLOAT_VECTOR, dim=768
            ),  # Adjust dim as needed
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(
            fields=fields, description="RAG collection for Milvus"
        )
        collection = Collection(name=args.collection_name, schema=schema)
        collection.create_index(
            "vector",
            {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
    else:
        collection = Collection(args.collection_name)
        collection.load()

    # Initialize generator
    generator = ChatGroqGenerator(
        model=args.llm_model,
        api_key=args.llm_api_key,
        llm_params=eval(args.llm_params),
    )

    # Load dataloader
    dataloader = TriviaQADataloader(
        answer_summary_generator=generator,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    dataloader.load_data()
    questions = dataloader.get_questions()

    # Initialize embeddings
    text_embedder = HuggingFaceEmbeddings(model_name=args.dense_model)
    FastEmbedSparse(model_name=args.sparse_model)

    # Initialize LLM
    llm = ChatGroq(
        model=args.llm_model,
        temperature=0,
        api_key=args.llm_api_key,
    )

    # Prepare prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Question: {question} Context: {context} Answer:""",
            ),
        ]
    )

    # Process each question
    for question in questions:
        dense_question_embedding = text_embedder.embed_query(question)

        # Query Milvus
        results = collection.search(
            data=[dense_question_embedding],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 128}},
            limit=args.top_k,
            expr=None,
        )

        # Extract content from query results
        retrieval_results = [
            hit.entity.get("content") for hits in results for hit in hits
        ]
        docs_content = "\n\n".join(retrieval_results)

        messages = prompt.invoke({"question": question, "context": docs_content})
        response = llm.invoke(messages)
        result = response.content
        print(f"Question: {question}\nAnswer: {result}\n")


if __name__ == "__main__":
    main()
