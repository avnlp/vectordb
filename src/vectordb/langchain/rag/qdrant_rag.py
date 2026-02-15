import argparse

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from qdrant_client import QdrantClient


def main():
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
        "--embedding_model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model.",
    )

    # LLM parameters
    parser.add_argument("--llm_api_key", required=True, help="API key for the LLM.")
    parser.add_argument(
        "--llm_model", default="llama-3.1-8b-instant", help="Model name for the LLM."
    )
    parser.add_argument(
        "--llm_max_tokens", type=int, default=512, help="Max tokens for LLM response."
    )

    # Prompt template
    parser.add_argument(
        "--prompt_template", type=str, help="Prompt template for the LLM."
    )

    args = parser.parse_args()

    # Initialize Qdrant VectorDB
    QdrantClient(url=args.qdrant_host, api_key=args.qdrant_api_key)

    # Initialize the generator
    generator = ChatGroqGenerator(
        model=args.generator_model,
        api_key=args.generator_api_key,
        llm_params=eval(args.generator_params),
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
