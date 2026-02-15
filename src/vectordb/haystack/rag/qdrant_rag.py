import argparse
from ast import literal_eval

from dataloaders.haystack import ARCDataloader,EdgarDataloader, FactScoreDataloader, PopQADataloader, TriviaQADataloader
from dataloaders.haystack.llms import ChatGroqGenerator
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from vectordb import QdrantDocumentConverter, QdrantVectorDB


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline with Qdrant and TriviaQA")

    # Qdrant parameters
    parser.add_argument("--qdrant_host", required=True, help="Qdrant host URL.")
    parser.add_argument("--qdrant_api_key", required=True, help="API key for Qdrant.")
    parser.add_argument("--collection_name", default="test_collection_dense1", help="Qdrant collection name.")

    # Generator parameters
    parser.add_argument("--generator_model", default="llama-3.1-8b-instant", help="Model for the ChatGroqGenerator.")
    parser.add_argument("--generator_api_key", required=True, help="API key for the generator.")
    parser.add_argument("--generator_params", default="{}", help="JSON string of additional LLM parameters.")

    # Dataloader parameters
    parser.add_argument("--dataset_name", required=True, help="Name of the TriviaQA dataset.")
    parser.add_argument("--split", default="test[:5]", help="Split of the dataset to use.")

    # Embedding model parameters
    parser.add_argument("--embedding_model", default="sentence-transformers/all-mpnet-base-v2", help="Embedding model.")

    # LLM parameters
    parser.add_argument("--llm_api_key", required=True, help="API key for the LLM.")
    parser.add_argument("--llm_model", default="llama-3.1-8b-instant", help="Model name for the LLM.")
    parser.add_argument("--llm_max_tokens", type=int, default=512, help="Max tokens for LLM response.")

    # Prompt template
    parser.add_argument("--prompt_template", type=str, help="Prompt template for the LLM.")

    args = parser.parse_args()

    # Initialize Qdrant VectorDB
    qdrant_client = QdrantClient(url=args.qdrant_host, api_key=args.qdrant_api_key)

    # Initialize the generator
    generator = ChatGroqGenerator(
        model=args.generator_model,
        api_key=args.generator_api_key,
        llm_params=eval(args.generator_params),
    )

    # Load data using TriviaQADataloader
    dataloader = TriviaQADataloader(
        answer_summary_generator=generator,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    dataloader.load_data()
    questions = dataloader.get_questions()
    dataloader.get_haystack_documents()

    # Initialize embedding model
    text_embedder = SentenceTransformersTextEmbedder(model=args.embedding_model)
    text_embedder.warm_up()

    # Initialize LLM
    llm = OpenAIGenerator(
        api_key=Secret.from_token(args.llm_api_key),
        api_base_url="https://api.groq.com/openai/v1",
        model=args.llm_model,
        generation_kwargs={"max_tokens": args.llm_max_tokens},
    )

    # Create RAG pipeline
    rag_pipeline = Pipeline()
    prompt_builder = PromptBuilder(template=args.prompt_template)
    answer_builder = AnswerBuilder()

    rag_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
    rag_pipeline.add_component(instance=llm, name="llm")
    rag_pipeline.add_component(instance=answer_builder, name="answer_builder")

    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")

    # Query Qdrant and process questions
    for question in questions:
        question_embedding = text_embedder.run(text=question)["embedding"]
        
        # Query Qdrant for relevant documents based on the question embedding
        query_response = qdrant_client.search(
            collection_name=args.collection_name,
            query_vector=question_embedding,
            limit=10,
            filter=Filter().must([{"key": "text", "match": {"any": ["Chipmunks"]}}]),  # Example filter on metadata
        )

        retrieval_results = QdrantDocumentConverter.convert_query_results_to_haystack_documents(query_response)

        result = rag_pipeline.run(
            data={
                "prompt_builder": {"question": question, "documents": retrieval_results},
                "answer_builder": {"query": question, "documents": retrieval_results},
            }
        )
        print(result)


if __name__ == "__main__":
    main()
