import argparse

from dataloaders import ARCDataloader, EdgarDataloader, FactScoreDataloader, PopQADataloader, TriviaQADataloader
from dataloaders.llms import ChatGroqGenerator
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import ServerlessSpec

from vectordb import PineconeDocumentConverter, PineconeVectorDB


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run data processing and upsert to Pinecone VectorDB.")

    # General Arguments
    parser.add_argument(
        "--dataloader",
        required=True,
        choices=["triviaqa", "arc", "popqa", "factscore", "edgar"],
        help="Choose the dataloader.",
    )
    parser.add_argument("--dataset_name", required=True, help="Dataset name for the chosen dataloader.")
    parser.add_argument("--split", default="test[:5]", help="Dataset split to use.")

    # Generator Arguments
    parser.add_argument("--llm_model", default="llama-3.1-8b-instant", help="Model name for the generator.")
    parser.add_argument("--llm_api_key", required=True, help="API key for the generator.")
    parser.add_argument(
        "--llm_params",
        type=str,
        default='{"temperature": 0, "max_tokens": 1024, "timeout": 360, "max_retries": 100}',
        help="JSON string of LLM parameters.",
    )

    # Embedding Arguments
    parser.add_argument(
        "--embedding_model", default="sentence-transformers/all-mpnet-base-v2", help="Embedding model name."
    )

    # Pinecone Arguments
    parser.add_argument("--pinecone_api_key", required=True, help="Pinecone API key.")
    parser.add_argument("--index_name", default="test1", help="Pinecone index name.")
    parser.add_argument("--dimension", type=int, default=768, help="Dimension of embeddings.")
    parser.add_argument("--metric", default="cosine", help="Metric for the Pinecone index.")
    parser.add_argument("--namespace", default="test_namespace", help="Namespace for upserting data into Pinecone.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for upserts.")
    parser.add_argument("--cloud", default="aws", help="Cloud provider for Pinecone (e.g., 'aws').")
    parser.add_argument("--region", default="us-east-1", help="Region for the Pinecone index.")

    args = parser.parse_args()

    # Initialize Generator
    generator = ChatGroqGenerator(
        model=args.llm_model,
        api_key=args.llm_api_key,
        llm_params=eval(args.llm_params),
    )

    # Map dataloader names to classes
    dataloader_map = {
        "triviaqa": TriviaQADataloader,
        "arc": ARCDataloader,
        "popqa": PopQADataloader,
        "factscore": FactScoreDataloader,
        "edgar": EdgarDataloader,
    }
    dataloader_cls = dataloader_map[args.dataloader]
    dataloader = dataloader_cls(dataset_name=args.dataset_name, split=args.split, answer_summary_generator=generator)

    # Load data
    dataloader.load_data()
    dataloader.get_questions()
    langchain_documents = dataloader.get_langchain_documents()

    # Initialize Embedding Model
    embedder = HuggingFaceEmbeddings(model_name=args.embedding_model)

    # Create embeddings
    texts = [doc.page_content for doc in langchain_documents]
    doc_embeddings = embedder.embed_documents(texts)

    # Initialize Pinecone VectorDB
    pinecone_vector_db = PineconeVectorDB(api_key=args.pinecone_api_key)
    pinecone_vector_db.create_index(
        index_name=args.index_name,
        dimension=args.dimension,
        metric=args.metric,
        spec=ServerlessSpec(cloud=args.cloud, region=args.region),
    )

    # Prepare data for Pinecone and upsert
    data_for_pinecone = PineconeDocumentConverter.prepare_langchain_documents_for_upsert(
        documents=langchain_documents, embeddings=doc_embeddings
    )
    pinecone_vector_db.upsert(
        data=data_for_pinecone, namespace=args.namespace, batch_size=args.batch_size, show_progress=True
    )


if __name__ == "__main__":
    main()
