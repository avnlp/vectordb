"""Qdrant RAG pipeline example with LangChain integration.

This module demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline
using Qdrant VectorDB and LangChain components.
"""

import argparse

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    """Run the Qdrant RAG pipeline.

    This function:
    - Parses command line arguments
    - Initializes Qdrant client and Pinecone VectorDB
    - Initializes the data loader and generator
    - Loads the TriviaQA dataset
    - Processes questions through the RAG pipeline
    - Generates answers using retrieved context
    """
    parser = argparse.ArgumentParser(
        description="RAG pipeline with Qdrant and TriviaQA"
    )

    # Qdrant parameters
    parser.add_argument("--qdrant_host", required=True, help="Qdrant host URL.")
    parser.add_argument("--qdrant_api_key", required=True, help="API key for Qdrant.")
    parser.add_argument(
        "--collection_name",
        default="test_collection_dense1",
        help="Qdrant collection name.",
    )

    # Generator parameters
    parser.add_argument(
        "--generator_model",
        default="llama-3.1-8b-instant",
        help="Model for the ChatGroqGenerator.",
    )
    parser.add_argument(
        "--generator_api_key", required=True, help="API key for the generator."
    )
    parser.add_argument(
        "--generator_params",
        default="{}",
        help="JSON string of additional LLM parameters.",
    )

    # Dataloader parameters
    parser.add_argument(
        "--dataset_name", required=True, help="Name of the TriviaQA dataset."
    )
    parser.add_argument(
        "--split", default="test[:5]", help="Split of the dataset to use."
    )

    # Embedding model parameters
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

    # LLM parameters
    parser.add_argument("--llm_api_key", required=True, help="API key for the LLM.")
    parser.add_argument(
        "--llm_model", default="llama-3.1-8b-instant", help="Model name for the LLM."
    )
    parser.add_argument(
        "--llm_max_tokens", type=int, default=512, help="Max tokens for LLM response."
    )
    parser.add_argument(
        "--llm_params",
        type=str,
        default='{"temperature": 0, "max_tokens": 1024, "timeout": 360, "max_retries": 100}',
        help="JSON string of LLM parameters.",
    )

    # Pinecone parameters
    parser.add_argument("--pinecone_api_key", required=True, help="Pinecone API key.")
    parser.add_argument(
        "--index_name", default="test-index-hybrid", help="Pinecone index name."
    )
    parser.add_argument(
        "--namespace", default="test_namespace1", help="Namespace for Pinecone queries."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to retrieve from Pinecone.",
    )

    # Prompt template
    parser.add_argument(
        "--prompt_template", type=str, help="Prompt template for the LLM."
    )

    args = parser.parse_args()

    # Initialize Qdrant client (for potential future use)
    QdrantClient(url=args.qdrant_host, api_key=args.qdrant_api_key)

    # Initialize Pinecone VectorDB
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
    )

    # Initialize the generator
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
    sparse_embedder = FastEmbedSparse(model_name=args.sparse_model)

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
                """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:""",
            ),
        ]
    )

    # Process each question
    for question in questions:
        dense_question_embedding = text_embedder.embed_query(question)
        sparse_question_embedding = sparse_embedder.embed_query(question)
        query_response = pinecone_vector_db.query(
            vector=dense_question_embedding,
            sparse_vector={
                "indices": sparse_question_embedding.indices,
                "values": sparse_question_embedding.values,
            },
            top_k=args.top_k,
            namespace=args.namespace,
        )
        retrieval_results = (
            PineconeDocumentConverter.convert_query_results_to_langchain_documents(
                query_response
            )
        )
        docs_content = "\n\n".join(doc.page_content for doc in retrieval_results)
        messages = prompt.invoke({"question": question, "context": docs_content})
        response = llm.invoke(messages)
        result = response.content
        print(f"Question: {question}\nAnswer: {result}\n")


if __name__ == "__main__":
    main()
