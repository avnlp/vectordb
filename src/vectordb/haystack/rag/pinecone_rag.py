import argparse

from dataloaders import TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main(args):
    # Initialize Pinecone Vector DB
    pinecone_vector_db = PineconeVectorDB(api_key=args.pinecone_api_key, index_name=args.index_name)

    # Initialize Generator
    generator = ChatGroqGenerator(
        model=args.llm_model,
        api_key=args.llm_api_key,
        llm_params={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
        },
    )

    # Load data
    dataloader = TriviaQADataloader(
        answer_summary_generator=generator,
        dataset_name=args.dataset_name,
        split=args.dataset_split,
    )
    dataloader.load_data()
    questions = dataloader.get_questions()
    dataloader.get_haystack_documents()

    # Initialize Text Embedder
    text_embedder = SentenceTransformersTextEmbedder(model=args.text_embedder_model)
    text_embedder.warm_up()

    # Build RAG Pipeline
    prompt_template = """Based on the Financial Article, answer the question.
    Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
    Question: {{question}}
    Answer:
    """

    llm = OpenAIGenerator(
        api_key=Secret.from_token(args.llm_api_key),
        api_base_url=args.llm_api_base_url,
        model=args.llm_model,
        generation_kwargs={"max_tokens": args.generation_max_tokens},
    )

    rag_pipeline = Pipeline()
    prompt_builder = PromptBuilder(template=prompt_template)
    answer_builder = AnswerBuilder()

    rag_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
    rag_pipeline.add_component(instance=llm, name="llm")
    rag_pipeline.add_component(instance=answer_builder, name="answer_builder")

    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")

    # Process Questions
    for question in questions:
        question_embedding = text_embedder.run(text=question)["embedding"]
        query_response = pinecone_vector_db.query(vector=question_embedding, namespace=args.namespace)
        retrieval_results = PineconeDocumentConverter.convert_query_results_to_haystack_documents(query_response)

        result = rag_pipeline.run(
            data={
                "prompt_builder": {"question": question, "documents": retrieval_results},
                "answer_builder": {"query": question, "documents": retrieval_results},
            }
        )
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Pipeline with TriviaQA and Pinecone")

    # Argument parser for user inputs
    parser = argparse.ArgumentParser(description="Script for processing and indexing data with Pinecone.")

    # Dataloader parameters
    parser.add_argument(
        "--dataloader",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Dataloader to use for loading datasets.",
    )
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset to be used by the dataloader.")
    parser.add_argument("--split", default="test", help="Dataset split to process (e.g., 'test', 'train').")
    parser.add_argument(
        "--text_splitter",
        default="RecursiveCharacterTextSplitter",
        help="Text splitter method to preprocess documents.",
    )
    parser.add_argument(
        "--text_splitter_params", type=str, help="JSON string of parameters for configuring the text splitter."
    )

    # Generator parameters
    parser.add_argument("--generator_model", type=str, help="Model name for the dataloader's generator.")
    parser.add_argument("--generator_api_key", help="API key for the dataloader generator.")
    parser.add_argument("--generator_llm_params", type=str, help="JSON string of parameters for the generator LLM.")

    # Embedder parameters
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model to use for generating document embeddings.",
    )
    parser.add_argument("--embedding_model_params", type=str, help="JSON string of parameters for the embedding model.")

    # Pinecone VectorDB parameters
    parser.add_argument("--api_key", required=True, help="API key for accessing Pinecone.")
    parser.add_argument("--host", help="Host URL for Pinecone.")
    parser.add_argument("--index_name", required=True, help="Name of the Pinecone index.")
    parser.add_argument("--proxy_url", help="Proxy URL for Pinecone.")
    parser.add_argument("--proxy_headers", type=str, help="JSON string of proxy headers for configuring Pinecone.")
    parser.add_argument("--ssl_ca_certs", help="Path to SSL CA certificates.")
    parser.add_argument("--ssl_verify", type=bool, default=True, help="Enable or disable SSL verification.")
    parser.add_argument(
        "--additional_headers", type=str, help="JSON string of additional headers for Pinecone requests."
    )
    parser.add_argument("--pool_threads", type=int, default=1, help="Number of threads for connection pooling.")
    parser.add_argument("--namespace", required=True, help="Namespace for data upserts in Pinecone.")
    parser.add_argument("--dimension", type=int, default=768, help="Vector dimension for the Pinecone index.")
    parser.add_argument("--metric", default="cosine", help="Similarity metric to use in the Pinecone index.")
    parser.add_argument("--cloud", default="aws", help="Cloud provider hosting the Pinecone database.")
    parser.add_argument("--region", default="us-east-1", help="Region where the Pinecone index is hosted.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for upserting data to Pinecone.")
    parser.add_argument("--show_progress", type=bool, default=True, help="Show progress bar while upserting data.")
    parser.add_argument("--tracing_project_name", type=str, help="Name of the tracing project.")
    parser.add_argument("--weave_params", type=str, help="JSON string of parameters for configuring Weave.")

    args = parser.parse_args()

    main(args)
