"""RAG pipeline implementation using Chroma vector database.

This module provides a Retrieval-Augmented Generation pipeline
using Chroma as the vector database with Haystack components.
"""

import argparse
from ast import literal_eval

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from vectordb import ChromaDocumentConverter, ChromaVectorDB


def main():
    """Run RAG pipeline using Chroma vector database.

    This function loads data, creates a RAG pipeline with retrieval
    from Chroma and LLM generation, and answers questions.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Run Chroma RAG Pipeline")

    # Chroma VectorDB arguments
    parser.add_argument(
        "--chroma_path",
        default="./chroma_database_files",
        help="Path for Chroma database files.",
    )
    parser.add_argument(
        "--chroma_collection",
        default="test_collection_dense1",
        help="Name of the Chroma collection.",
    )
    parser.add_argument("--vector_db_path", required=True, help="Path to Chroma .")

    # Dataloader Arguments

    # Generator arguments
    parser.add_argument(
        "--generator_model",
        type=str,
        required=True,
        help="Model name for the dataloader's generator.",
    )
    parser.add_argument(
        "--generator_api_key",
        required=True,
        help="API key for the dataloader generator.",
    )
    parser.add_argument(
        "--generator_llm_params",
        type=str,
        help="JSON string of parameters for the generator LLM.",
    )

    # Query arguments
    parser.add_argument("--dataset_name", required=True, help="Dataset Name")
    parser.add_argument(
        "--n_results",
        type=int,
        default=10,
        help="Number of results to retrieve from the database.",
    )
    parser.add_argument(
        "--tracing_project_name",
        default="chroma",
        help="Name of the Weave project for tracing.",
    )
    parser.add_argument(
        "--weave_params",
        type=str,
        help="JSON string of parameters for initializing Weave.",
    )

    args = parser.parse_args()

    # Initialize Chroma VectorDB
    weave_params = literal_eval(args.weave_params) if args.weave_params else {}
    chroma_vector_db = ChromaVectorDB(
        persistent=True,
        path=args.chroma_path,
        tracing_project_name=args.tracing_project_name,
        weave_params=weave_params,
    )
    chroma_vector_db = ChromaVectorDB(path=args.chroma_path)
    chroma_vector_db.create_collection(name=args.chroma_collection)

    # Initialize the LLM Generator
    generator = ChatGroqGenerator(
        model=args.model_name,
        api_key=args.api_key,
        llm_params={
            "temperature": 0,
            "max_tokens": 1024,
            "timeout": 360,
            "max_retries": 100,
        },
    )

    # Load dataset
    dataloader = TriviaQADataloader(
        answer_summary_generator=generator,
        dataset_name=args.dataset_name,
        split="test[:5]",
    )

    questions = dataloader.get_questions()

    # Initialize text embedder
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-mpnet-base-v2"
    )
    text_embedder.warm_up()

    # Build a LLM Pipeline to answer questions based on Semantic Search Results
    prompt_template = """Based on the Financial Article, answer the question.
    Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
    Question: {{question}}
    Answer:
    """

    llm = OpenAIGenerator(
        api_key=Secret.from_token(args.api_key),
        api_base_url="https://api.groq.com/openai/v1",
        model=args.model_name,
        generation_kwargs={"max_tokens": 512},
    )

    rag_pipeline = Pipeline()
    prompt_builder = PromptBuilder(template=prompt_template)
    answer_builder = AnswerBuilder()

    rag_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
    rag_pipeline.add_component(instance=llm, name="llm")
    rag_pipeline.add_component(instance=answer_builder, name="answer_builder")

    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")

    chroma_vector_db = ChromaVectorDB()

    for question in questions:
        # Generate dense embedding for the query
        question_embedding = text_embedder.run(text=question)["embedding"]

        query_response = chroma_vector_db.query(
            vector=question_embedding, n_results=args.n_results
        )
        retrieval_results = (
            ChromaDocumentConverter.convert_query_results_to_haystack_documents(
                query_response
            )
        )

        result = rag_pipeline.run(
            data={
                "prompt_builder": {
                    "question": question,
                    "documents": retrieval_results,
                },
                "answer_builder": {"query": question, "documents": retrieval_results},
            }
        )
        print(result)


if __name__ == "__main__":
    main()
