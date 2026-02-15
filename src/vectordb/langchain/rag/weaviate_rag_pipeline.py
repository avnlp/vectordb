"""Weaviate RAG pipeline example with LangChain integration.

This module demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline
using Weaviate VectorDB and LangChain components.
"""

import argparse

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    """Run the Weaviate RAG pipeline.

    This function:
    - Parses command line arguments
    - Initializes Weaviate VectorDB and LLM generator
    - Loads the TriviaQA dataset
    - Processes questions through the RAG pipeline
    - Generates answers using retrieved context from Weaviate
    """
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="RAG pipeline for question answering using Pinecone and ChatGroq."
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

    # Pinecone arguments
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

    args = parser.parse_args()

    # Initialize Pinecone
    pinecone_vector_db = PineconeVectorDB(
        api_key=args.pinecone_api_key,
        index_name=args.index_name,
    )

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
