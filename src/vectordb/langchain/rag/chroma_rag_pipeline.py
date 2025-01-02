import argparse

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from vectordb import ChromaDocumentConverter, PineconeVectorDB


def main():
    parser = argparse.ArgumentParser(description="Run Chroma RAG Pipeline with Dense Embedding Search.")

    # Chroma VectorDB arguments
    parser.add_argument("--chroma_path", default="./chroma_database_files", help="Path for Chroma database files.")
    parser.add_argument("--chroma_collection", default="test_collection_dense1", help="Name of the Chroma collection.")
    parser.add_argument("--vector_db_path", required=True, help="Path to Chroma or Pinecone vector database files.")

    # LLM arguments
    parser.add_argument(
        "--model_name", required=True, help="Model for generating summaries (e.g., 'llama-3.1-8b-instant')."
    )
    parser.add_argument("--api_key", required=True, help="API key for the LLM service.")

    # Query arguments
    parser.add_argument("--dataset_name", required=True, help="Dataset name for Trivia QA.")
    parser.add_argument("--n_results", type=int, default=10, help="Number of results to retrieve from the database.")

    args = parser.parse_args()

    # Initialize Chroma VectorDB
    chroma_vector_db = ChromaVectorDB(persistent=True, path=args.chroma_path)
    chroma_vector_db.create_collection(name=args.chroma_collection)

    # Initialize the LLM Generator
    generator = ChatGroqGenerator(
        model=args.model_name,
        api_key=args.api_key,
        llm_params={"temperature": 0, "max_tokens": 1024, "timeout": 360, "max_retries": 100},
    )

    # Load dataset
    dataloader = TriviaQADataloader(
        answer_summary_generator=generator,
        dataset_name=args.dataset_name,
        split="test[:5]",
    )

    questions = dataloader.get_questions()

    # Initialize text embedder
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
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

    pinecone_vector_db = PineconeVectorDB()  # Assuming Pinecone is set up

    for question in questions:
        # Generate dense embedding for the query
        question_embedding = text_embedder.run(text=question)["embedding"]

        # Query the vector database (Chroma or Pinecone)
        query_response = pinecone_vector_db.query(vector=question_embedding, n_results=args.n_results)
        retrieval_results = ChromaDocumentConverter.convert_query_results_to_haystack_documents(query_response)

        result = rag_pipeline.run(
            data={
                "prompt_builder": {"question": question, "documents": retrieval_results},
                "answer_builder": {"query": question, "documents": retrieval_results},
            }
        )
        print(result)


if __name__ == "__main__":
    main()
