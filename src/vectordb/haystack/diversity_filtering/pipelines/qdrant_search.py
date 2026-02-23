"""Qdrant search pipeline with diversity filtering and RAG.

Implements search with Maximum Margin Relevance (MMR) diversity filtering
for Qdrant vector database, with optional RAG answer generation.

Pipeline Flow:
1. Embed query using SentenceTransformersTextEmbedder
2. Retrieve top_k_candidates from Qdrant using vector similarity
3. Apply MMR diversity ranking to select diverse subset
4. Optionally generate RAG answer using diverse documents

Diversity Filtering:
Uses SentenceTransformersDiversityRanker with configurable similarity metric
(cosine or dot_product) and top_k parameter. The MMR algorithm balances
query relevance against inter-document diversity.

RAG Generation:
When enabled, formats diverse documents using dataset-specific prompts and
generates answers via OpenAIGenerator (Groq or OpenAI providers).

Configuration:
Pipeline behavior controlled via YAML config with Qdrant connection params
(url, api_key), retrieval settings, diversity algorithm options, and RAG
settings. Supports both local and cloud Qdrant deployments.
"""

from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersDiversityRanker

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.diversity_filtering.utils.config_loader import (
    ConfigLoader,
)
from vectordb.haystack.diversity_filtering.utils.prompts import (
    format_documents,
    get_prompt_template,
)


def run_search(config_path: str, query: str) -> dict:
    """Run Qdrant search pipeline with diversity filtering and optional RAG.

    Args:
        config_path: Path to configuration YAML file.
        query: Search query string.

    Returns:
        Dictionary with search results including diverse documents and optional answer.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If configuration invalid.
    """
    config = ConfigLoader.load(config_path)

    embedder = SentenceTransformersTextEmbedder(
        model=config.embedding.model,
        device=config.embedding.device,
    )
    embedder.warm_up()

    query_embedding = embedder.run(text=query)["embedding"]

    db = QdrantVectorDB(
        url=config.vectordb.qdrant.url,
        api_key=config.vectordb.qdrant.api_key,
        index=config.index.name,
        embedding_dim=config.embedding.dimension,
    )

    candidates = db.search(
        query_embedding=query_embedding,
        top_k=config.retrieval.top_k_candidates,
    )

    if not candidates:
        return {
            "documents": [],
            "num_diverse": 0,
            "answer": None,
            "query": query,
        }

    if config.diversity.algorithm == "maximum_margin_relevance":
        ranker = SentenceTransformersDiversityRanker(
            model=config.embedding.model,
            top_k=config.diversity.top_k,
            similarity="cosine"
            if config.diversity.similarity_metric == "cosine"
            else "dot_product",
        )
        diverse_docs = ranker.run(documents=candidates, query=query)["documents"]
    else:
        diverse_docs = candidates[: config.diversity.top_k]

    result = {
        "documents": [
            {
                "content": doc.content,
                "meta": doc.meta,
                "score": getattr(doc, "score", None),
            }
            for doc in diverse_docs
        ],
        "num_diverse": len(diverse_docs),
        "query": query,
        "answer": None,
    }

    if config.rag.enabled:
        try:
            doc_content = format_documents(
                [{"content": doc.content, "meta": doc.meta} for doc in diverse_docs]
            )

            prompt_template = get_prompt_template(config.dataset.name)

            prompt_builder = PromptBuilder(template=prompt_template)
            prompt = prompt_builder.run(query=query, documents=doc_content)["prompt"]

            generator = OpenAIGenerator(
                api_key_env_var=f"{config.rag.provider.upper()}_API_KEY",
                model=config.rag.model,
                generation_kwargs={
                    "temperature": config.rag.temperature,
                    "max_tokens": config.rag.max_tokens,
                },
            )
            response = generator.run(prompt=prompt)
            result["answer"] = response.get("replies", [None])[0]

        except Exception as e:
            result["answer"] = f"Error generating answer: {str(e)}"

    return result
